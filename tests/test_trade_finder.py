"""Tests for the M-HATE trade engine: lineup math, each penalty, redundancy rules, tiers, generation."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trade_finder import (  # noqa: E402
    FALLBACK_DRV, TAP_RATE, TSP_TARGET_RECEIVES_FEWER, TSP_TARGET_RECEIVES_MORE, TSP_TWO_FOR_TWO,
    LeagueSnapshot, PlayerValue, TeamSnapshot, TradeContext, TradeProposal, build_slots, generate_trades,
    lineup_ppg, single_slot_positions, slot_context, structure_penalty, value_over_replacement,
)

SLOTS = build_slots({"QB": 1, "RB": 2, "WR": 2, "TE": 1, "RB/WR/TE": 1, "D/ST": 1, "K": 1, "BE": 6})
WEEKS = 4
CTX = slot_context(SLOTS, FALLBACK_DRV)


def P(name, pos, pts):
    return PlayerValue(name, pos, "X", "ACTIVE", pts, [pts] * WEEKS)


def core(prefix="", extra=(), qb=20, te=8):
    return [P(prefix + "QB", "QB", qb), P(prefix + "TE", "TE", te), P(prefix + "RB1", "RB", 15),
            P(prefix + "RB2", "RB", 12), P(prefix + "WR1", "WR", 14), P(prefix + "WR2", "WR", 12),
            P(prefix + "DST", "D/ST", 7), P(prefix + "K", "K", 7)] + list(extra)


def snapshot(me, target, filler=()):
    teams = {1: TeamSnapshot(1, "Me", me), 2: TeamSnapshot(2, "Them", target)}
    for i, roster in enumerate(filler, start=3):
        teams[i] = TeamSnapshot(i, f"Team {i}", roster)
    return LeagueSnapshot(teams, SLOTS, list(range(5, 5 + WEEKS)), True)


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

def test_vacant_starter_slot_falls_back_to_waiver_value():
    roster = core(extra=[P("FlexRB", "RB", 9)], te=12)
    lost = weekly(roster) - weekly(without(roster, "TE"))
    assert abs(lost - (12 - FALLBACK_DRV["TE"])) < 1e-9


def test_vacant_starter_slot_uses_best_bench_player_when_better_than_waiver():
    roster = core(extra=[P("BenchTE", "TE", 8.5), P("FlexRB", "RB", 9)], te=12)
    assert abs((weekly(roster) - weekly(without(roster, "TE"))) - (12 - 8.5)) < 1e-9


def test_opponent_losing_a_starter_and_bench_player_is_replaced_from_what_remains():
    roster = core(extra=[P("Watson", "WR", 16.8), P("Montgomery", "RB", 12.7), P("BenchWR", "WR", 10.5)])
    assert abs((weekly(roster) - weekly(without(roster, "Watson", "Montgomery"))) - 7.0) < 1e-9


def test_one_qb_one_te_format_detection():
    assert single_slot_positions(SLOTS) == {"QB", "TE"}
    superflex = build_slots({"QB": 1, "RB": 2, "WR": 2, "TE": 1, "OP": 1})
    assert single_slot_positions(superflex) == {"TE"}


# ---- target penalties

def test_structure_penalty_table():
    assert structure_penalty(1, 1) == 0.0
    assert structure_penalty(2, 2) == TSP_TWO_FOR_TWO
    assert structure_penalty(2, 1) == TSP_TARGET_RECEIVES_MORE      # target receives 2, gives 1
    assert structure_penalty(1, 2) == TSP_TARGET_RECEIVES_FEWER     # target receives 1, gives 2


def pair(me_extra, target_extra, filler=(), **kw):
    return snapshot(core("m", me_extra, **kw), core("t", target_extra), filler)


def test_tsp_applied_to_trades():
    snap = pair([P("a", "WR", 13), P("b", "WR", 12.5)], [P("x", "WR", 13), P("y", "WR", 12.5)])
    assert evaluate(snap, ["a"], ["x"]).tsp == 0.0
    assert evaluate(snap, ["a", "b"], ["x", "y"]).tsp == TSP_TWO_FOR_TWO
    assert evaluate(snap, ["a", "b"], ["x"]).tsp == TSP_TARGET_RECEIVES_MORE
    assert evaluate(snap, ["a"], ["x", "y"]).tsp == TSP_TARGET_RECEIVES_FEWER


def test_bcp_only_when_target_receives_two_plus_and_cuts_a_bench_player():
    bench = P("tbench", "WR", 11.5)
    snap = pair([P("a", "WR", 13), P("b", "WR", 12.5)], [P("x", "WR", 13), bench])
    t = evaluate(snap, ["a", "b"], ["x"])
    assert t.theirs.drops == [bench]
    assert abs(t.bcp - 0.5 * value_over_replacement(bench, FALLBACK_DRV)) < 1e-9
    assert evaluate(snap, ["a"], ["x"]).bcp == 0.0


def test_tap_applies_when_target_surrenders_an_s_tier_asset():
    fillers = [core(f"f{i}", [P(f"fw{i}", "WR", 10 + i * 0.1)]) for i in range(3)]
    snap = pair([P("a", "WR", 20)], [P("Star", "WR", 22)], fillers)
    assert abs(evaluate(snap, ["a"], ["Star"]).tap - TAP_RATE * 22) < 1e-9
    assert evaluate(snap, ["a"], ["tRB2"]).tap == 0.0


def test_plt_when_target_surrenders_qb1_without_a_starting_replacement():
    snap = pair([P("BenchQB", "QB", 14)], [], qb=17)
    # target's QB1 (tQB, 20) goes away; my bench QB (14) comes back but can't be tQB's replacement in a
    # way that starts? it does start (nobody else), so no PLT
    t = evaluate(snap, ["BenchQB"], ["tQB"])
    assert t.plt == 0.0
    # target gets a non-QB back: PLT = QB1 PPG - waiver DRV
    snap2 = pair([P("aWR", "WR", 13)], [], qb=17)
    t2 = evaluate(snap2, ["aWR"], ["tQB"])
    assert abs(t2.plt - PLT_QB_TE_RATE * (20 - FALLBACK_DRV["QB"])) < 1e-9


def test_mai_is_delta_lineup_minus_the_penalties():
    snap = pair([P("a", "WR", 13), P("b", "WR", 12.5)], [P("x", "WR", 13)])
    t = evaluate(snap, ["a", "b"], ["x"])
    assert abs(t.mai - (t.delta_lineup - t.tap - t.bcp - t.tsp - t.plt)) < 1e-9
    assert abs(t.nmu - (t.my_gain + t.mai)) < 1e-9


# ---- redundancy (your side and theirs)

def test_second_qb_behind_a_top_tier_qb1_contributes_nothing():
    # my QB1 (mQB, 20) is the best QB in the league -> S-tier; an incoming better QB adds 0
    snap = pair([P("aWR", "WR", 13)], [P("StarQB2", "QB", 25)], qb=20)
    t = evaluate(snap, ["aWR"], ["StarQB2"])
    assert t.mine.delta <= 0 or all(p.name != "StarQB2" for p, _, _ in t.mine.upgrades)


def test_no_starting_depth_for_a_redundant_bench_piece():
    snap = pair([P("aWR", "WR", 14.5)], [P("BackupQB", "QB", 15)], qb=25)
    t = evaluate(snap, ["aWR"], ["BackupQB"])
    assert t.mine.redundant and t.mine.blocked and t.tier is None


# ---- tiers and generation

def test_generated_trades_meet_each_tier_bar_on_both_sides():
    me = core("m", [P("mWR3", "WR", 13), P("mRB3", "RB", 14), P("mRB4", "RB", 12.5)])
    target = core("t", [P("tWR3", "WR", 15), P("tWR4", "WR", 13.5), P("tTE2", "TE", 6.0)])
    result = generate_trades(snapshot(me, target), 1, 2)
    for tier, proposals in result.items():
        for t in proposals:
            assert t.tier == tier and not t.blocked
            assert t.theirs.delta >= 0 and t.my_gain > 0
            assert t.you_why and t.them_why
            if tier == "Win-Win":
                assert t.mai >= 2.0 and t.my_gain >= 1.5 and (len(t.send), len(t.receive)) in ((1, 1), (2, 2))
            if tier == "Worth a Shot":
                assert t.mai >= 0.5 and t.my_gain >= 1.0
            if tier == "Long Shot":
                assert -1.5 <= t.mai < 0.5 and t.my_gain >= 3.0


def test_strict_mode_only_returns_trades_the_target_clearly_clears():
    me = core("m", [P("mWR3", "WR", 13), P("mRB3", "RB", 14), P("mRB4", "RB", 12.5)])
    target = core("t", [P("tWR3", "WR", 15), P("tWR4", "WR", 13.5), P("tTE2", "TE", 6.0)])
    result = generate_trades(snapshot(me, target), 1, 2, strict=True)
    assert all(t.mai >= 1.5 for ps in result.values() for t in ps)


# ---- calibration against real outcomes (7 of 7 multi-player offers rejected, 1-for-1s entertained)

from trade_finder import (  # noqa: E402
    AP_PENALTY, LATERAL_TAX, PLT_QB_TE_RATE, QB_GAIN_HAIRCUT, TAP_PACKAGE_MULT, VOID_RATE,
)


def league_with(me_extra, target_extra, me_kw=None, target_kw=None, star_league=True):
    """Two teams plus mediocre filler so a 22+ PPG player is S-tier"""
    fillers = [core(f"f{i}", [P(f"fw{i}", "WR", 10 + i * 0.1)]) for i in range(4)] if star_league else []
    return snapshot(core("m", me_extra, **(me_kw or {})), core("t", target_extra, **(target_kw or {})), fillers)


def test_one_for_one_cross_position_trade_that_fills_both_holes_is_win_win():
    # Dobbins-for-Worthy / Saquon-for-Davante shape: simple, balanced, cross-positional
    me = core("m", [P("mRB3", "RB", 13), P("mRB4", "RB", 12.5)])
    for p in me:
        if p.name == "mRB2":
            p.weekly[:], p.base = [14.0] * WEEKS, 14.0     # RBs 15/14 + flex 13 start, RB4 sits
        if p.name == "mWR2":
            p.weekly[:], p.base = [9.0] * WEEKS, 9.0       # my hole
    target = core("t", [P("tWR3", "WR", 11.5)])
    for p in target:
        if p.name == "tRB2":
            p.weekly[:], p.base = [8.0] * WEEKS, 8.0       # their hole
    fillers = [core(f"f{i}", [P(f"fw{i}", "WR", 10 + i * 0.1)]) for i in range(4)]
    t = evaluate(snapshot(me, target, fillers), ["mRB4"], ["tWR3"])
    assert t.tsp == 0.0 and t.lat == 0.0 and t.tap == 0.0
    assert t.mine.delta >= 1.5 and t.mai >= 2.0 and t.tier == "Win-Win"


def test_depth_package_for_a_tier_one_asset_is_taxed_double():
    snap = league_with([P("mRB3", "RB", 14), P("mRB4", "RB", 13)], [P("Star", "WR", 22)])
    t = evaluate(snap, ["mRB3", "mRB4"], ["Star"])          # Stevenson + Olave -> A.J. Brown
    assert abs(t.tap - TAP_RATE * 22 * TAP_PACKAGE_MULT) < 1e-9
    assert t.tier != "Win-Win"


def test_asking_two_alpha_assets_for_one_player_is_not_offered():
    snap = league_with([P("mAllen", "QB", 24)], [P("Chase", "WR", 22), P("Mitchell", "RB", 23)], me_kw={"qb": 21})
    t = evaluate(snap, ["mAllen"], ["Chase", "Mitchell"])   # Allen -> Chase + Mitchell
    assert t.tap > 8 and t.tier is None


def test_lateral_same_position_swap_pays_the_why_bother_tax():
    snap = league_with([P("mRB3", "RB", 13), P("mRB4", "RB", 12.5)], [P("tRB3", "RB", 13), P("tRB4", "RB", 12.5)])
    t = evaluate(snap, ["mRB3", "mRB4"], ["tRB3", "tRB4"])  # Skattebo + Dobbins -> Warren + Mitchell
    assert t.lat == LATERAL_TAX and t.tier != "Win-Win"
    assert evaluate(snap, ["mRB3"], ["tRB3"]).lat == LATERAL_TAX


def test_surrendering_a_premier_wr_without_a_wr_back_leaves_a_taxed_hole():
    snap = league_with([P("mQB2", "QB", 20)], [P("Waddle", "WR", 22)], star_league=True)
    t = evaluate(snap, ["mQB2"], ["Waddle"])                # Kyler + LaPorta -> Waddle + Likely shape
    assert t.plt >= VOID_RATE * (22 - FALLBACK_DRV["WR"]) - 1e-9


def test_asymmetry_when_target_gives_two_starters_for_one_starter_plus_a_bench_piece():
    snap = league_with([P("Good", "WR", 18), P("Scrub", "WR", 8.0)], [P("T1", "WR", 15), P("T2", "WR", 14.5)])
    t = evaluate(snap, ["Good", "Scrub"], ["T1", "T2"])     # Douglas + Waddle -> Harrison + Etienne shape
    assert t.ap == AP_PENALTY and t.tier != "Win-Win"


def test_two_for_one_and_two_for_two_never_reach_win_win():
    snap = league_with([P("mRB3", "RB", 14), P("mRB4", "RB", 13)], [P("tWR3", "WR", 14), P("tWR4", "WR", 13)])
    for send, recv in ((["mRB3", "mRB4"], ["tWR3"]), (["mRB3", "mRB4"], ["tWR3", "tWR4"]), (["mRB3"], ["tWR3", "tWR4"])):
        assert evaluate(snap, send, recv).tier != "Win-Win"
    assert evaluate(snap, ["mRB3", "mRB4"], ["tWR3"]).tsp == TSP_TARGET_RECEIVES_MORE


def test_qb_upgrades_count_half_for_the_target_in_one_qb_leagues():
    snap = league_with([P("mQB2", "QB", 21)], [], target_kw={"qb": 14})
    t = evaluate(snap, ["mQB2"], ["tRB2"])
    qb_gain = 21 - 14
    assert abs((t.theirs.delta - t.theirs.perceived) - QB_GAIN_HAIRCUT * qb_gain) < 0.75
    assert t.theirs.perceived < t.theirs.delta


# ---- talent floor, candidate pools and replacement level

from trade_finder import TALENT_FLOOR_RATIO, TALENT_GAP_RATE, explain, fetch_drv  # noqa: E402


def test_talent_floor_penalizes_trading_a_clearly_better_player_for_a_lesser_one():
    snap = league_with([P("mWR3", "WR", 12.0)], [P("tWR3", "WR", 15.0)], star_league=False)
    t = evaluate(snap, ["mWR3"], ["tWR3"])           # target hands over the 15, gets the 12
    assert 12.0 < TALENT_FLOOR_RATIO * 15.0
    assert abs(t.tfl - TALENT_GAP_RATE * (15.0 - 12.0)) < 1e-9
    even = evaluate(snap, ["mWR2"], ["tWR2"])         # 12 for 12
    assert even.tfl == 0.0


def test_generation_never_picks_kickers_defenses_or_ir_players():
    me = core("m", [P("mWR3", "WR", 13), P("mIR", "RB", 18)])
    for p in me:
        if p.name == "mIR":
            p.injury = "INJURY_RESERVE"
    target = core("t", [P("tWR3", "WR", 15)])
    result = generate_trades(snapshot(me, target), 1, 2)
    for proposals in result.values():
        for t in proposals:
            for p in t.send + t.receive:
                assert p.position not in ("K", "D/ST") and p.injury != "INJURY_RESERVE"


class _FA:
    def __init__(self, pts, status="ACTIVE"):
        self.avg_points, self.projected_avg_points, self.total_points, self.injuryStatus = pts, pts, pts * 4, status


class _League:
    def __init__(self, pts_by_pos):
        self.pts_by_pos = pts_by_pos

    def free_agents(self, week=None, size=25, position=None):
        return [_FA(v, s) for v, s in self.pts_by_pos[position]]


def test_replacement_level_averages_the_top_three_healthy_free_agents():
    pts = {pos: [(1.0, "ACTIVE")] for pos in FALLBACK_DRV}
    pts["WR"] = [(30.0, "OUT"), (12.0, "ACTIVE"), (10.0, "ACTIVE"), (8.0, "ACTIVE"), (2.0, "ACTIVE")]
    drv, live, sources = fetch_drv(_League(pts), 5)
    assert abs(drv["WR"] - 10.0) < 1e-9 and live
    assert [round(f[1]) for f in sources["WR"]] == [12, 10, 8]


# ---- effective structure and bench-level pieces (feedback on the Bowers+Corum and Kyler trades)

def test_a_throw_in_below_waiver_level_does_not_count_as_a_player_in_the_structure():
    # Bowers + Corum for Wilson + Montgomery: Corum wouldn't beat a waiver RB, so it's really 1-for-2
    me = core("m", [P("Corum", "RB", 8.2)], te=16.2)
    target = core("t", [P("Wilson", "WR", 13.3), P("Montgomery", "RB", 12.7)], te=11.8)
    fillers = [core(f"f{i}", te=13) for i in range(3)]      # so a 11.8 TE isn't S-tier
    t = evaluate(snapshot(me, target, fillers), ["mTE", "Corum"], ["Wilson", "Montgomery"])
    assert (len(t.send), len(t.receive)) == (2, 2)
    assert (t.real_in, t.real_out) == (1, 2)
    assert t.tsp == TSP_TARGET_RECEIVES_FEWER
    assert "1-for-2" in explain(t)[1]


def test_a_backup_qb_who_only_covers_a_bye_adds_nothing_for_the_target():
    # target has a real QB1 (bye in week 2); the incoming 15.8 QB would start only that week
    me = core("m", [P("Kyler", "QB", 15.8)])
    target = core("t", [P("Waddle", "WR", 12.2)])
    for p in target:
        if p.name == "tQB":
            p.weekly[:] = [20.0, 0.0, 20.0, 20.0]
    t = evaluate(snapshot(me, target), ["Kyler"], ["Waddle"])
    assert not t.theirs.upgrades
    assert t.theirs.delta <= 1e-9          # they only lose Waddle; the backup QB contributes nothing


def test_a_waiver_filled_slot_names_the_rostered_player_the_model_is_bypassing():
    # their rostered QB (12.0) is below the waiver level (13.0), so the model assumes they'd stream:
    # the explanation has to say so, instead of just "filled from waivers"
    me = core("m", [P("Kyler", "QB", 16.0)])
    target = core("t", [P("Waddle", "WR", 12.0)], qb=12.0)
    t = evaluate(snapshot(me, target), ["Kyler"], ["Waddle"])
    assert t.theirs.upgrades and t.theirs.upgrades[0][1] is None
    them_why = explain(t)[1]
    assert "tQB" in them_why and "waivers" in them_why


def test_a_bench_player_above_waiver_level_is_still_a_throw_in_if_he_would_not_start():
    # live waiver RB level is 6.3 in this league, so Corum (8.2) beats waivers, but he sits behind
    # the target's real RBs; only Bowers is a real piece for them: 1-for-2, not 2-for-2
    me = core("m", [P("Corum", "RB", 8.2)], te=16.2)
    target = core("t", [P("Wilson", "WR", 13.3), P("Montgomery", "RB", 12.7)], te=11.8)
    snap = snapshot(me, target, [core(f"f{i}", te=13) for i in range(3)])
    snap.drv["RB"] = 6.3
    t = evaluate(snap, ["mTE", "Corum"], ["Wilson", "Montgomery"])
    assert (t.real_in, t.real_out) == (1, 2)
    assert t.tsp == TSP_TARGET_RECEIVES_FEWER
    assert [p.name for p in t.real_in_players] == ["mTE"]
    assert t.tier is None or t.mai < 0.5      # no longer a Worth a Shot


# ---- explaining why a custom trade misses every tier

from trade_finder import why_no_tier  # noqa: E402


def _proposal(mine_delta, theirs_delta, blocked=False, **penalties):
    from trade_finder import SideImpact
    return TradeProposal(
        target_id=2, target_name="Them", send=[], receive=[],
        mine=SideImpact(delta=mine_delta, perceived=mine_delta, blocked=blocked),
        theirs=SideImpact(delta=theirs_delta, perceived=theirs_delta),
        tap=penalties.get("tap", 0.0), bcp=0.0, tsp=penalties.get("tsp", 0.0), plt=0.0,
    )


def test_why_no_tier_names_the_bar_that_was_missed():
    # Bowers-for-Watson shape: easy for them (MAI +2.1) but only +0.4 for you
    msg = why_no_tier(_proposal(0.4, 2.1))
    assert "+2.1" in msg and "+0.4" in msg and "1.0" in msg and "Worth a Shot" in msg
    assert "hurts" in why_no_tier(_proposal(-1.0, 3.0)) or "doesn't improve your lineup" in why_no_tier(_proposal(-1.0, 3.0))
    assert "gets worse" in why_no_tier(_proposal(2.0, -0.5))
    assert "floor" in why_no_tier(_proposal(4.0, 1.0, tsp=3.5))          # MAI -2.5
    assert "Long Shot" in why_no_tier(_proposal(2.0, 0.3))               # MAI +0.3, needs 3.0+
    assert "Blocked" in why_no_tier(_proposal(2.0, 2.0, blocked=True))


# ---- lineup before/after (so the math can be checked by eye)

from trade_finder import lineup_rows  # noqa: E402


def test_backup_te_steps_in_when_the_starter_is_traded_and_the_lineup_table_shows_it():
    me = core("m", [P("Loveland", "TE", 11.0), P("Addison", "WR", 10.4)], te=16.2)
    target = core("t", [P("Watson", "WR", 16.8)], te=11.8)
    fillers = [core(f"f{i}", te=13) for i in range(3)]
    snap = snapshot(me, target, fillers)
    tc = TradeContext(snap, 1, 2)
    t = tc.evaluate([p for p in me if p.name == "mTE"], [p for p in target if p.name == "Watson"])
    tc.attach_lineups(t)
    before = {slot_who: pts for slot, slot_who, pts in t.mine.lineup_before if slot == "TE"}
    after = {slot_who: pts for slot, slot_who, pts in t.mine.lineup_after if slot == "TE"}
    assert before == {"mTE": 16.2}
    assert after == {"Loveland": 11.0}       # Loveland plays TE, not a waiver pickup
    # so losing Bowers costs 16.2 - 11.0 at TE, not 16.2 - the waiver TE
    assert abs((11.0 - 16.2) - (-5.2)) < 1e-9


def test_blocked_message_names_the_side_and_the_players():
    # Loveland (bench TE) for a starting WR: THEY would be swapping a starter for a redundant bench TE
    me = core("m", [P("Loveland", "TE", 7.3)], te=16.2)
    target = core("t", [P("Tate", "WR", 11.2)], te=15.0)
    fillers = [core(f"f{i}", te=9) for i in range(3)]
    tc = TradeContext(snapshot(me, target, fillers), 1, 2)
    t = tc.evaluate([p for p in me if p.name == "Loveland"], [p for p in target if p.name == "Tate"])
    assert t.blocked and t.theirs.blocked and not t.mine.blocked
    msg = why_no_tier(t)
    assert msg.startswith("Blocked: They'd give up starter Tate for bench piece Loveland at TE")


# ---- explaining an empty result

def test_empty_result_explains_itself_and_offers_the_closest_misses():
    # a fair swap that only nudges your lineup: no tier, but it should show up as a near miss
    me = core("m", [P("mRB3", "RB", 13)])
    for p in me:
        if p.name == "mWR2":
            p.weekly[:], p.base = [9.5] * WEEKS, 9.5
    target = core("t", [P("tWR3", "WR", 10.5)])
    for p in target:
        if p.name == "tRB2":
            p.weekly[:], p.base = [8.0] * WEEKS, 8.0
    result = generate_trades(snapshot(me, target, [core(f"f{i}", te=13) for i in range(3)]), 1, 2)
    assert result.funnel["checked"] > 0
    assert sum(result.funnel[k] for k in result.funnel if k != "checked") == result.funnel["checked"]
    for t in result.near_misses:
        assert t.tier is None and not t.blocked and t.theirs.delta >= 0 and t.mine.delta > 0
        assert t.you_why and why_no_tier(t)


def test_lowering_the_minimum_gain_lets_a_small_but_fair_trade_through():
    me = core("m", [P("mRB3", "RB", 13)])
    for p in me:
        if p.name == "mWR2":
            p.weekly[:], p.base = [9.5] * WEEKS, 9.5
    target = core("t", [P("tWR3", "WR", 10.5)])
    for p in target:
        if p.name == "tRB2":
            p.weekly[:], p.base = [8.0] * WEEKS, 8.0
    snap = snapshot(me, target, [core(f"f{i}", te=13) for i in range(3)])
    strict_bar = generate_trades(snap, 1, 2)                       # default 1.0
    lower = generate_trades(snap, 1, 2, min_gain=0.1)
    assert sum(len(v) for v in lower.values()) >= sum(len(v) for v in strict_bar.values())
    assert all(t.my_gain >= 0.1 for ps in lower.values() for t in ps)


# ---- selling low: a spare elite player shouldn't be given away (Josh Allen for Mike Evans)

def test_selling_an_elite_spare_qb_for_a_lesser_wr_is_not_recommended():
    me = core("m", [P("Allen", "QB", 25.9)], qb=22.4)                 # two elite QBs, one QB slot
    for p in me:
        if p.name == "mWR2":
            p.weekly[:], p.base = [9.0] * WEEKS, 9.0
    target = core("t", [P("Evans", "WR", 13.6)], qb=16.1)
    fillers = [core(f"f{i}", qb=17 + i) for i in range(3)]
    tc = TradeContext(snapshot(me, target, fillers), 1, 2)
    t = tc.evaluate([p for p in me if p.name == "Allen"], [p for p in target if p.name == "Evans"])
    assert t.mine.delta > -1.0                        # the spare QB barely matters to your lineup...
    assert t.sell_low > 5.0                           # ...but you'd be giving away 25.9 for 13.6
    assert t.my_effective_gain < 0 and t.tier is None
    assert "selling" in why_no_tier(t) and "Allen" in why_no_tier(t)
    result = generate_trades(snapshot(me, target, fillers), 1, 2)
    assert all(
        not (len(x.send) == 1 and x.send[0].name == "Allen" and x.receive[0].name == "Evans")
        for group in list(result.values()) + [result.near_misses] for x in group
    )


def test_an_even_talent_swap_has_no_sell_low_penalty():
    snap = pair([P("a", "WR", 13)], [P("x", "WR", 12)])
    assert evaluate(snap, ["a"], ["x"]).sell_low == 0.0
