"""Tests for the trade engine: lineup math, asset value, the acceptance model, tiers, generation,
and the edge cases that are meant to expose flaws rather than confirm the engine works."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trade_finder import (  # noqa: E402
    CONVEXITY, ENDOWMENT, FALLBACK_DRV, HASSLE, RAW_POINTS_ANCHOR, VALUE_WEIGHT, LeagueSnapshot, PlayerValue, TeamSnapshot,
    TradeContext, asset_value, breakdown, build_slots, fetch_drv, generate_trades, lineup_ppg,
    single_slot_positions, slot_context, starter_levels, wanted_players, why_no_tier,
)

SLOTS = build_slots({"QB": 1, "RB": 2, "WR": 2, "TE": 1, "RB/WR/TE": 1, "D/ST": 1, "K": 1, "BE": 6})
WEEKS = 4
CTX = slot_context(SLOTS, FALLBACK_DRV)


def P(name, pos, pts, injury="ACTIVE", weekly=None):
    return PlayerValue(name, pos, "X", injury, pts, list(weekly) if weekly else [pts] * WEEKS)


def core(prefix="", extra=(), qb=20, te=8, rb=(15, 12), wr=(14, 12)):
    return [P(prefix + "QB", "QB", qb), P(prefix + "TE", "TE", te), P(prefix + "RB1", "RB", rb[0]),
            P(prefix + "RB2", "RB", rb[1]), P(prefix + "WR1", "WR", wr[0]), P(prefix + "WR2", "WR", wr[1]),
            P(prefix + "DST", "D/ST", 7), P(prefix + "K", "K", 7)] + list(extra)


def fillers(n=8):
    """A realistic middle of the league, so last-starter levels are sane (10 teams by default)"""
    return [core(f"f{i}", [P(f"f{i}RB3", "RB", 9 + i * 0.2), P(f"f{i}WR3", "WR", 10 + i * 0.2),
                           P(f"f{i}QB2", "QB", 12 + i * 0.3)], qb=16 + i * 0.5, te=7 + i * 0.5)
            for i in range(n)]


def snapshot(me, target, filler=None):
    teams = {1: TeamSnapshot(1, "Me", me), 2: TeamSnapshot(2, "Them", target)}
    for i, roster in enumerate(fillers() if filler is None else filler, start=3):
        teams[i] = TeamSnapshot(i, f"Team {i}", roster)
    return LeagueSnapshot(teams, SLOTS, list(range(5, 5 + WEEKS)), True, drv_live=True)


def evaluate(snap, send_names, recv_names):
    tc = TradeContext(snap, 1, 2)
    send = [p for p in snap.teams[1].players if p.name in send_names]
    recv = [p for p in snap.teams[2].players if p.name in recv_names]
    return tc.evaluate(send, recv)


def weekly(players):
    return lineup_ppg(players, CTX, WEEKS)


def without(players, *names):
    return [p for p in players if p.name not in names]


# ---- lineup math

def test_rostered_players_start_over_a_better_waiver_pickup():
    # a 12.0 QB starts even though the waiver QB is 13.0: waivers don't bench rostered starters
    assert abs((weekly(core(qb=13.0)) - weekly(core(qb=12.0))) - 1.0) < 1e-9


def test_a_slot_nobody_can_fill_falls_back_to_the_waiver_value():
    roster = core(te=12)
    assert abs((weekly(roster) - weekly(without(roster, "TE"))) - (12 - FALLBACK_DRV["TE"])) < 1e-9


def test_a_bench_player_steps_in_when_a_starter_leaves_even_if_below_waiver_level():
    roster = core(extra=[P("BenchTE", "TE", 5.0)], te=12)        # waiver TE is 6.5, bench TE 5.0 still plays
    assert abs((weekly(roster) - weekly(without(roster, "TE"))) - (12 - 5.0)) < 1e-9


def test_a_player_far_below_waiver_level_is_streamed_over():
    # nobody starts a 5.0 TE in the flex when a 9.5 waiver RB is sitting there
    roster = core(extra=[P("BenchTE", "TE", 5.0)], te=12)
    assert abs(weekly(roster) - weekly(core(te=12))) < 1e-9      # the flex is a waiver pickup either way


def test_bye_and_out_weeks_are_filled_by_the_next_player_or_waivers():
    qb = P("QB", "QB", 20, weekly=[20, 0, 20, 20])
    roster = [qb] + core()[1:]
    assert abs(weekly(roster) - (weekly(core()) - (20 - FALLBACK_DRV["QB"]) / WEEKS)) < 1e-9


def test_one_qb_one_te_format_detection():
    assert single_slot_positions(SLOTS) == {"QB", "TE"}
    superflex = build_slots({"QB": 1, "RB": 2, "WR": 2, "TE": 1, "OP": 1})
    assert single_slot_positions(superflex) == {"TE"}


# ---- asset value

def test_last_starter_levels_scale_with_league_size_and_format():
    snap = snapshot(core("m"), core("t"))
    levels = starter_levels(snap.teams, SLOTS)
    qbs = sorted((p.base for t in snap.teams.values() for p in t.players if p.position == "QB"), reverse=True)
    assert levels["QB"] == qbs[9]                                  # 10 teams, 1 QB each
    # 10 x (1 + 0.1 flex share) = 11 TEs are "starters" but only 10 are rostered: the last starter is a free agent
    assert levels["TE"] == max(FALLBACK_DRV["TE"], min(p.base for t in snap.teams.values() for p in t.players if p.position == "TE"))
    superflex = build_slots({"QB": 1, "RB": 2, "WR": 2, "TE": 1, "OP": 1})
    assert starter_levels(snap.teams, superflex)["QB"] < levels["QB"]   # superflex: QBs are scarcer


def test_a_star_is_worth_more_than_two_average_starters():
    levels = {"WR": 10.0}
    star, mid = P("Star", "WR", 22), P("Mid", "WR", 16)
    assert asset_value(star, levels) > 2 * asset_value(mid, levels)
    v = 16 - (1 - RAW_POINTS_ANCHOR) * 10.0
    assert abs(asset_value(mid, levels) - v * (1 + v / CONVEXITY)) < 1e-9


def test_a_long_injury_discounts_asset_value_but_a_questionable_tag_barely_does():
    levels = {"RB": 8.0}
    assert asset_value(P("IR", "RB", 18, "INJURY_RESERVE"), levels) <= 0.5 * asset_value(P("Ok", "RB", 18), levels)
    assert asset_value(P("Q", "RB", 18, "QUESTIONABLE"), levels) > 0.98 * asset_value(P("Ok", "RB", 18), levels)


def test_replacement_level_averages_the_top_three_healthy_free_agents():
    class _FA:
        def __init__(self, pts, status):
            self.avg_points, self.projected_avg_points, self.total_points = pts, pts, pts * 4
            self.injuryStatus, self.name = status, f"fa{pts}"

    class _League:
        def __init__(self, pts_by_pos):
            self.pts_by_pos = pts_by_pos

        def free_agents(self, week=None, size=25, position=None):
            return [_FA(v, s) for v, s in self.pts_by_pos[position]]

    pts = {pos: [(1.0, "ACTIVE")] for pos in FALLBACK_DRV}
    pts["WR"] = [(30.0, "OUT"), (12.0, "ACTIVE"), (10.0, "ACTIVE"), (8.0, "ACTIVE"), (2.0, "ACTIVE")]
    drv, live, sources = fetch_drv(_League(pts), 5)
    assert abs(drv["WR"] - 10.0) < 1e-9 and live
    assert [round(f[1]) for f in sources["WR"]] == [12, 10, 8]


# ---- the acceptance model, one charge per objection

def test_a_one_for_one_has_no_friction_and_a_package_has_exactly_one():
    snap = snapshot(core("m", [P("a", "RB", 13), P("b", "RB", 12.5)]), core("t", [P("x", "WR", 13), P("y", "WR", 12.5)]))
    assert evaluate(snap, ["a"], ["x"]).theirs.friction == 0.0
    assert evaluate(snap, ["a", "b"], ["x"]).theirs.hassle == HASSLE[3]
    assert evaluate(snap, ["a", "b"], ["x", "y"]).theirs.hassle == HASSLE[4]
    assert all(label in ("value lost", "package hassle") for label, _ in evaluate(snap, ["a", "b"], ["x"]).penalties)


def test_a_sideways_same_position_swap_is_charged_once_for_not_being_worth_the_bother():
    snap = snapshot(core("m", [P("a", "WR", 13)]), core("t", [P("x", "WR", 12.5)]))
    t = evaluate(snap, ["a"], ["x"])
    assert t.theirs.lateral > 0 and t.theirs.hassle == 0.0
    assert "sideways" in t.fails_because


def test_the_target_overvalues_what_it_gives_up():
    snap = snapshot(core("m", [P("a", "WR", 16)]), core("t", [P("x", "WR", 16)]))
    t = evaluate(snap, ["a"], ["x"])
    assert abs(t.mine.value_delta) < 1e-9                         # same player, rational view
    assert abs(t.theirs.value_delta - (1 - ENDOWMENT) * t.theirs.value_out) < 1e-9


def test_acceptance_is_lineup_plus_value_minus_friction():
    snap = snapshot(core("m", [P("a", "WR", 13), P("b", "WR", 12.5)]), core("t", [P("x", "WR", 16)]))
    t = evaluate(snap, ["a", "b"], ["x"])
    th = t.theirs
    assert abs(t.acceptance - (th.lineup_delta + VALUE_WEIGHT * th.value_delta - th.hassle + th.freed)) < 1e-9
    assert abs(t.nmu - (t.my_effective_gain + t.acceptance)) < 1e-9


# ---- edge cases (meant to expose flaws)

def test_1_elite_player_for_two_mediocre_players_is_rejected():
    me = core("m", [P("mRB3", "RB", 14), P("mRB4", "RB", 13)])
    target = core("t", [P("Star", "WR", 22)])
    t = evaluate(snapshot(me, target), ["mRB3", "mRB4"], ["Star"])
    assert t.theirs.shape in ("diversification", "depth")
    assert t.theirs.value_delta < 0 and t.acceptance < -3 and t.tier is None
    assert "best player in the deal" in t.fails_because


def test_2_two_good_starters_for_one_elite_starter_is_attractive_consolidation():
    me = core("m", [P("Star", "WR", 22)], wr=(14, 12))
    target = core("t", [P("tRB3", "RB", 14), P("tRB4", "RB", 13)], wr=(13, 11))
    t = evaluate(snapshot(me, target), ["Star"], ["tRB3", "tRB4"])
    assert t.theirs.shape == "consolidation" and t.theirs.freed > 0
    assert t.theirs.lineup_delta > 0 and t.theirs.value_delta > 0 and t.acceptance > 3
    # ...and it costs me the best player, which the engine says plainly instead of proposing it
    assert t.mine.total < 0 and t.tier is None
    assert t.mine.best_out.name == "Star" and "Star" in why_no_tier(t)


def test_2b_a_two_for_one_is_proposed_when_the_star_fills_my_holes_and_they_consolidate():
    me = core("m", [P("mRB3", "RB", 15), P("mRB4", "RB", 14)], rb=(16, 15), wr=(13, 11))
    target = core("t", [P("Star", "WR", 20)], rb=(12, 6), wr=(20, 14))
    target = [p for p in target if p.name != "tWR1"]
    t = evaluate(snapshot(me, target), ["mRB3", "mRB4"], ["Star"])
    assert t.theirs.lineup_delta > 2 and t.theirs.shape == "diversification"
    assert t.tier == "Long Shot"                                  # they still give up the best player


def test_3_a_starter_for_redundant_bench_depth_is_blocked():
    me = core("m", [P("BackupQB", "QB", 15)], qb=24)
    target = core("t", [P("Tate", "WR", 13)], qb=23)
    t = evaluate(snapshot(me, target), ["BackupQB"], ["Tate"])
    assert t.theirs.redundant and t.theirs.blocked and t.tier is None
    assert why_no_tier(t).startswith("Blocked: They'd give up starter Tate for BackupQB")


def test_4_a_player_at_waiver_level_has_no_asset_value_and_wins_nothing():
    me = core("m", [P("Scrub", "RB", 8.5)])
    target = core("t", [P("Decent", "WR", 13), P("tWR4", "WR", 11)])
    snap = snapshot(me, target)
    t = evaluate(snap, ["Scrub"], ["Decent"])
    assert asset_value(t.send[0], TradeContext(snap, 1, 2).levels) < 2.0
    assert t.theirs.redundant == [t.send[0]] and t.tier is None


def test_5_qb_for_qb_when_the_target_can_stream_is_a_lateral_nobody_bothers_with():
    me = core("m", qb=16.5)
    target = core("t", qb=15.0)
    t = evaluate(snapshot(me, target), ["mQB"], ["tQB"])
    assert 0 < t.theirs.lineup_delta < 2 and t.tier is None     # +1.5 at QB isn't worth their time
    assert t.theirs.lateral > 0 and t.acceptance < 2.0


def test_5b_a_real_qb_upgrade_over_the_targets_own_qb_counts_in_full():
    # Maye-for-Likely shape: their Herbert (13.9) starts over the 15.1 waiver QB, so Maye (15.9) is +2 to them
    me = core("m", [P("Maye", "QB", 15.9)], qb=16.3, te=9.3)
    target = core("t", [P("Likely", "TE", 12.8), P("Kincaid", "TE", 8.7)], qb=13.9)
    target = [p for p in target if p.name != "tTE"]
    snap = snapshot(me, target)
    snap.drv["QB"] = 15.1
    t = evaluate(snap, ["Maye"], ["Likely"])
    qb_gain = next(g for p, _, g in t.theirs.upgrades if p.name == "Maye")
    assert abs(qb_gain - 2.0) < 1e-6
    assert t.theirs.lineup_delta < 0                              # they still lose more at TE than they gain at QB
    assert t.mine.lineup_delta > 2.5
    assert "Likely" in t.fails_because                            # the engine says what they'd be giving up


def test_6_an_elite_te_for_multiple_pieces_is_a_star_for_depth():
    me = core("m", [P("mWR3", "WR", 14), P("mRB3", "RB", 13)], te=7)
    target = core("t", te=16)
    snap = snapshot(me, target)
    t = evaluate(snap, ["mWR3", "mRB3"], ["tTE"])
    levels = TradeContext(snap, 1, 2).levels
    assert asset_value(t.receive[0], levels) > asset_value(t.send[0], levels) + asset_value(t.send[1], levels)
    assert t.tier is None and t.acceptance < 0


def test_7_two_for_two_where_both_sides_improve_is_worth_a_shot_not_win_win():
    me = core("m", [P("mRB3", "RB", 14), P("mRB4", "RB", 13.5)], rb=(16, 15), wr=(11, 10))
    target = core("t", [P("tWR3", "WR", 14), P("tWR4", "WR", 13)], rb=(10, 9), wr=(16, 15))
    t = evaluate(snapshot(me, target), ["mRB3", "mRB4"], ["tWR3", "tWR4"])
    assert t.mine.lineup_delta > 0 and t.theirs.lineup_delta > 0
    assert t.tier == "Worth a Shot"                               # 4-player deals never reach Win-Win


def test_8_filling_a_major_lineup_hole_is_a_win_win():
    me = core("m", [P("mRB3", "RB", 13), P("mRB4", "RB", 12.5)], wr=(14, 8))         # weak WR2
    target = core("t", [P("tWR3", "WR", 12.5), P("tWR4", "WR", 11)], rb=(15, 8))      # weak RB2
    t = evaluate(snapshot(me, target), ["mRB3"], ["tWR3"])
    assert "RB" in t.theirs.need_solved and "WR" in t.mine.need_solved
    assert t.tier == "Win-Win"


def test_9_both_improve_but_one_side_loses_its_best_player():
    # their WR1 (21) for my RB (15) + WR (14): their lineup improves, but they give up the best player
    me = core("m", [P("mRB3", "RB", 15), P("mWR3", "WR", 14)], wr=(12, 11))
    target = core("t", [P("Alpha", "WR", 21), P("tWR3", "WR", 10)], rb=(11, 6), wr=(21, 11))
    target = [p for p in target if p.name != "tWR1"]
    t = evaluate(snapshot(me, target), ["mRB3", "mWR3"], ["Alpha"])
    assert t.theirs.lineup_delta > 0
    assert t.theirs.value_delta < 0 and "best player in the deal" in t.fails_because
    assert t.acceptance < t.theirs.lineup_delta                   # value loss and hassle pull it down


def test_10_fair_by_ppg_but_bad_by_roster_construction():
    # a 14-PPG RB for their 14-PPG WR2: fair on paper, but their RBs are 16/15/14.5 so the RB sits
    me = core("m", [P("mRB3", "RB", 14)])
    target = core("t", [P("tRB3", "RB", 14.5), P("tWR3", "WR", 11)], rb=(16, 15), wr=(16, 14))
    t = evaluate(snapshot(me, target), ["mRB3"], ["tWR2"])
    assert t.theirs.start_share.get(id(t.send[0]), 0.0) == 0.0 and t.theirs.redundant
    assert t.theirs.lineup_delta < -2 and t.tier is None


def test_11_slightly_unfair_by_ppg_but_very_attractive_by_roster_construction():
    # a 12.5-PPG RB for a 13.5-PPG WR: they "lose" a point of PPG, but their RB2 is 7 and WR3 is 13
    me = core("m", [P("mRB3", "RB", 12.5), P("mRB4", "RB", 12)], wr=(14, 10))
    target = core("t", [P("tWR3", "WR", 13.5), P("tWR4", "WR", 12)], rb=(15, 7), wr=(16, 14))
    t = evaluate(snapshot(me, target), ["mRB3"], ["tWR3"])
    assert t.send[0].base < t.receive[0].base
    assert t.theirs.lineup_delta > 3 and t.acceptance > 1.5 and t.tier == "Win-Win"


def test_12_an_injured_player_counts_less_now_and_is_flagged():
    healthy = core("m", [P("Back", "RB", 15)])
    hurt = core("m", [P("Back", "RB", 15, "OUT", weekly=[0, 4.5, 9, 12.75])])
    target = core("t", [P("tWR3", "WR", 13)], rb=(15, 8))
    t_ok = evaluate(snapshot(healthy, target), ["Back"], ["tWR3"])
    t_hurt = evaluate(snapshot(hurt, target), ["Back"], ["tWR3"])
    assert t_hurt.theirs.lineup_delta < t_ok.theirs.lineup_delta
    assert t_hurt.theirs.value_in < t_ok.theirs.value_in          # a long injury also dents asset value
    assert any("Out" in r for r in t_hurt.theirs.risks) and t_hurt.confidence != "high"


def test_13_a_brutal_schedule_moves_the_lineup_number_but_not_the_asset_value():
    easy = core("m", [P("Back", "RB", 15, weekly=[17, 17, 17, 17])])
    hard = core("m", [P("Back", "RB", 15, weekly=[12, 12, 12, 12])])
    target = core("t", [P("tWR3", "WR", 13)], rb=(15, 8))
    t_easy = evaluate(snapshot(easy, target), ["Back"], ["tWR3"])
    t_hard = evaluate(snapshot(hard, target), ["Back"], ["tWR3"])
    assert t_easy.theirs.lineup_delta > t_hard.theirs.lineup_delta
    assert abs(t_easy.theirs.value_in - t_hard.theirs.value_in) < 1e-9
    assert t_easy.acceptance - t_hard.acceptance <= 5.0 + 1e-9    # bounded by the lineup window, not amplified


def test_14_rb_heavy_team_consolidating_into_a_wr_is_attractive():
    me = core("m", [P("Star", "WR", 20)], wr=(15, 13))
    target = core("t", [P("tRB3", "RB", 14), P("tRB4", "RB", 13.5)], rb=(15, 14.5), wr=(12, 10))
    t = evaluate(snapshot(me, target), ["Star"], ["tRB3", "tRB4"])
    assert t.theirs.shape == "consolidation" and t.theirs.lineup_delta > 3 and t.acceptance > 3
    # attractive for THEM; the engine still won't propose it because I'd be the one giving the star
    assert t.mine.total < 0 and t.tier is None and "Star" in why_no_tier(t)


def test_15_wr_heavy_team_receiving_rb_depth_it_cannot_start_is_unattractive():
    me = core("m", [P("mRB3", "RB", 11), P("mRB4", "RB", 10.5)])
    target = core("t", [P("tWR3", "WR", 15)], rb=(16, 15), wr=(17, 16))
    t = evaluate(snapshot(me, target), ["mRB3", "mRB4"], ["tWR3"])
    assert t.theirs.blocked or t.acceptance < -3
    assert t.tier is None


# ---- your side: sell-low protection comes from asset value, not a separate rule

def test_selling_an_elite_spare_qb_for_a_lesser_wr_is_not_recommended():
    me = core("m", [P("Allen", "QB", 25.9)], qb=22.4, wr=(14, 9))
    target = core("t", [P("Evans", "WR", 13.6)], qb=16.1)
    snap = snapshot(me, target)
    t = evaluate(snap, ["Allen"], ["Evans"])
    assert t.mine.lineup_delta > 0                                 # Evans starts over my WR2
    assert t.mine.value_delta < 0 and t.my_effective_gain < 0.3 and t.tier is None
    assert "Allen" in why_no_tier(t)
    result = generate_trades(snap, 1, 2)
    assert all(
        not (len(x.send) == 1 and x.send[0].name == "Allen" and len(x.receive) == 1 and x.receive[0].name == "Evans")
        for group in list(result.values()) + [result.near_misses] for x in group
    )


# ---- tiers and generation

def test_generated_trades_meet_each_tier_bar_on_both_sides():
    me = core("m", [P("mWR3", "WR", 13), P("mRB3", "RB", 14), P("mRB4", "RB", 12.5)], wr=(14, 9))
    target = core("t", [P("tWR3", "WR", 15), P("tWR4", "WR", 13.5), P("tTE2", "TE", 6.0)], rb=(15, 8))
    result = generate_trades(snapshot(me, target), 1, 2)
    assert sum(len(v) for v in result.values()) > 0
    for tier, proposals in result.items():
        for t in proposals:
            assert t.tier == tier and not t.blocked
            assert t.my_effective_gain > 0 and t.you_why and t.them_why and t.mine.lineup_before
            if tier == "Win-Win":
                assert t.acceptance >= 1.5 and t.my_effective_gain >= 1.5 and t.simple
            if tier == "Worth a Shot":
                assert t.acceptance >= 0.5 and t.my_effective_gain >= 0.3
            if tier == "Long Shot":
                assert -3.0 <= t.acceptance < 0.5 and t.my_effective_gain >= 2.0


def test_generation_starts_from_what_each_manager_would_start():
    me = core("m", [P("mRB3", "RB", 14), P("Dud", "RB", 7)])
    target = core("t", rb=(15, 8))
    tc = TradeContext(snapshot(me, target), 1, 2)
    names = [p.name for p in wanted_players(tc, tc.theirs, tc.mine.team.players)]
    assert "mRB3" in names and "Dud" not in names and "mK" not in names and "mDST" not in names


def test_strict_mode_only_returns_trades_the_target_clearly_clears():
    me = core("m", [P("mWR3", "WR", 13), P("mRB3", "RB", 14), P("mRB4", "RB", 12.5)], wr=(14, 9))
    target = core("t", [P("tWR3", "WR", 15), P("tWR4", "WR", 13.5)], rb=(15, 8))
    result = generate_trades(snapshot(me, target), 1, 2, strict=True)
    assert all(t.acceptance >= 1.5 for ps in result.values() for t in ps)


def test_empty_result_explains_itself_and_offers_the_closest_misses():
    me = core("m", [P("mRB3", "RB", 13)], wr=(14, 9.5))
    target = core("t", [P("tWR3", "WR", 10.5)], rb=(15, 8))
    result = generate_trades(snapshot(me, target), 1, 2)
    assert result.funnel["checked"] > 0
    assert sum(result.funnel[k] for k in result.funnel if k != "checked") == result.funnel["checked"]
    for t in result.near_misses:
        assert t.tier is None and not t.blocked and t.my_effective_gain >= 0.2 and t.acceptance >= -3.0
        assert t.you_why and why_no_tier(t)


def test_lowering_the_minimum_gain_lets_a_small_but_fair_trade_through():
    me = core("m", [P("mRB3", "RB", 13)], wr=(14, 9.5))
    target = core("t", [P("tWR3", "WR", 10.5)], rb=(15, 8))
    snap = snapshot(me, target)
    lower = generate_trades(snap, 1, 2, min_gain=0.1)
    assert sum(len(v) for v in lower.values()) >= sum(len(v) for v in generate_trades(snap, 1, 2).values())
    assert all(t.my_effective_gain >= 0.1 for ps in lower.values() for t in ps)


def test_every_proposal_carries_a_full_quality_breakdown():
    me = core("m", [P("mRB3", "RB", 13)], wr=(14, 9))
    target = core("t", [P("tWR3", "WR", 12.5)], rb=(15, 8))
    t = evaluate(snapshot(me, target), ["mRB3"], ["tWR3"])
    labels = [k for k, _ in breakdown(t.theirs, True)]
    for needed in ("Lineup", "Value in / out", "Shape", "Need solved", "Best player out", "Total"):
        assert needed in labels
    assert t.confidence in ("high", "medium", "low") and t.works_because and t.fails_because


def test_the_targets_spare_qb_is_not_valued_like_a_starter():
    # the classic deal: they have two starting QBs, I need one, I send an RB that starts for them
    me = core("m", [P("mRB3", "RB", 9.5)], qb=15.9)
    target = core("t", [P("Stafford", "QB", 16.5), P("tWR3", "WR", 11)], qb=18.0, rb=(14, 6.7))
    t = evaluate(snapshot(me, target), ["mRB3"], ["Stafford"])
    assert t.theirs.best_out_share < 0.5                          # Stafford rode their bench
    assert t.theirs.lineup_delta > 0 and t.acceptance > 0.5 and t.tier is not None
    assert "backup" in t.them_why or "bench" in t.fails_because or t.theirs.value_delta >= 0


def test_the_targets_starting_qb_still_counts_in_full():
    me = core("m", [P("mRB3", "RB", 9.5)], qb=15.9)
    target = core("t", rb=(14, 6.7), qb=18.0)
    t = evaluate(snapshot(me, target), ["mRB3"], ["tQB"])
    assert t.theirs.lineup_delta < 0 and t.acceptance < -2 and t.tier is None
