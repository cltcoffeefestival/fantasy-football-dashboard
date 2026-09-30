"""Tests for the MAI trade engine: lineup math, each penalty, tiers and generation."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trade_finder import (  # noqa: E402
    AP_PENALTY, FALLBACK_DRV, LONG_SHOT_MIN, TAP_RATE, TSP_RECEIVES_MORE, TSP_TWO_FOR_TWO,
    LeagueSnapshot, PlayerValue, TeamSnapshot, TradeContext, build_slots, generate_trades,
    lineup_ppg, slot_context, tier_for, value_over_replacement,
)

SLOTS = build_slots({"QB": 1, "RB": 2, "WR": 2, "TE": 1, "RB/WR/TE": 1, "D/ST": 1, "K": 1, "BE": 6})
WEEKS = 4
CTX = slot_context(SLOTS, FALLBACK_DRV)


def P(name, pos, pts):
    return PlayerValue(name, pos, "X", "ACTIVE", pts, [pts] * WEEKS)


def core(prefix="", extra=()):
    return [P(prefix + "QB", "QB", 20), P(prefix + "RB1", "RB", 15), P(prefix + "RB2", "RB", 12),
            P(prefix + "WR1", "WR", 14), P(prefix + "WR2", "WR", 12), P(prefix + "DST", "D/ST", 7),
            P(prefix + "K", "K", 7)] + list(extra)


def snapshot(me, target, filler=()):
    teams = {1: TeamSnapshot(1, "Me", me), 2: TeamSnapshot(2, "Them", target)}
    for i, roster in enumerate(filler, start=3):
        teams[i] = TeamSnapshot(i, f"Team {i}", roster)
    return LeagueSnapshot(teams, SLOTS, list(range(5, 5 + WEEKS)), True)


def weekly(players):
    return lineup_ppg(players, CTX, WEEKS)


def without(players, *names):
    return [p for p in players if p.name not in names]


# ---- lineup math (dLineup)

def test_vacant_starter_slot_falls_back_to_waiver_value():
    roster = core(extra=[P("Bowers", "TE", 12), P("FlexRB", "RB", 9)])
    lost = weekly(roster) - weekly(without(roster, "Bowers"))
    assert abs(lost - (12 - FALLBACK_DRV["TE"])) < 1e-9


def test_vacant_starter_slot_uses_best_bench_player_when_better_than_waiver():
    roster = core(extra=[P("Bowers", "TE", 12), P("BenchTE", "TE", 8.5), P("FlexRB", "RB", 9)])
    lost = weekly(roster) - weekly(without(roster, "Bowers"))
    assert abs(lost - (12 - 8.5)) < 1e-9


def test_opponent_losing_a_starter_and_a_bench_player_is_replaced_from_what_remains():
    roster = core(extra=[P("TE", "TE", 8), P("Watson", "WR", 16.8), P("Montgomery", "RB", 12.7), P("BenchWR", "WR", 10.5)])
    after = without(roster, "Watson", "Montgomery")
    # RB 12.7 -> 12 (-0.7), WR slots 16.8/14 -> 14/12 (-4.8), flex 12 -> 10.5 (-1.5)
    assert abs((weekly(roster) - weekly(after)) - 7.0) < 1e-9


def test_replacement_level_comes_from_the_snapshot_drv():
    roster = core(extra=[P("Bowers", "TE", 12)])
    high = slot_context(SLOTS, {**FALLBACK_DRV, "TE": 9.0})
    assert lineup_ppg(without(roster, "Bowers"), high, WEEKS) > lineup_ppg(without(roster, "Bowers"), CTX, WEEKS)


# ---- penalties

def pair(me_extra, target_extra, filler=()):
    return snapshot(core("m", me_extra), core("t", target_extra), filler)


def evaluate(snap, send_names, recv_names):
    tc = TradeContext(snap, 1, 2)
    send = [p for p in snap.teams[1].players if p.name in send_names]
    recv = [p for p in snap.teams[2].players if p.name in recv_names]
    return tc.evaluate(send, recv)


def test_tsp_by_structure():
    snap = pair([P("a", "WR", 13), P("b", "WR", 12.5)], [P("x", "WR", 13), P("y", "WR", 12.5)])
    assert evaluate(snap, ["a"], ["x"]).tsp == 0.0
    assert evaluate(snap, ["a", "b"], ["x", "y"]).tsp == TSP_TWO_FOR_TWO
    assert evaluate(snap, ["a", "b"], ["x"]).tsp == TSP_RECEIVES_MORE
    assert evaluate(snap, ["a"], ["x", "y"]).tsp == 0.0


def test_bcp_is_half_the_value_over_replacement_of_the_bench_player_they_cut():
    bench = P("tbench", "WR", 11.5)
    snap = pair([P("a", "WR", 13), P("b", "WR", 12.5)], [P("x", "WR", 13), bench])
    t = evaluate(snap, ["a", "b"], ["x"])
    assert t.target_drops == [bench]
    assert abs(t.bcp - 0.5 * value_over_replacement(bench, FALLBACK_DRV)) < 1e-9
    assert evaluate(snap, ["a"], ["x"]).bcp == 0.0


def test_tap_applies_when_the_target_surrenders_an_s_tier_asset():
    star = P("Star", "WR", 22)
    fillers = [core(f"f{i}", [P(f"fw{i}", "WR", 10 + i * 0.1)]) for i in range(3)]
    snap = pair([P("a", "WR", 20)], [star], fillers)
    t = evaluate(snap, ["a"], ["Star"])
    assert abs(t.tap - TAP_RATE * 22) < 1e-9
    assert evaluate(snap, ["a"], ["tRB2"]).tap == 0.0


def test_ap_when_target_surrenders_two_starters_for_one_starter_plus_bench():
    me = [P("Good", "WR", 18), P("Scrub", "WR", 8.0)]
    target = [P("T1", "WR", 15), P("T2", "WR", 14.5)]
    snap = pair(me, target)
    t = evaluate(snap, ["Good", "Scrub"], ["T1", "T2"])
    assert t.ap == AP_PENALTY
    # send two starters instead: no asymmetry
    snap2 = pair([P("Good", "WR", 18), P("Good2", "WR", 17)], target)
    assert evaluate(snap2, ["Good", "Good2"], ["T1", "T2"]).ap == 0.0


def test_mai_is_delta_lineup_minus_the_penalties():
    snap = pair([P("a", "WR", 13), P("b", "WR", 12.5)], [P("x", "WR", 13)])
    t = evaluate(snap, ["a", "b"], ["x"])
    assert abs(t.mai - (t.delta_lineup - t.tap - t.bcp - t.tsp - t.ap)) < 1e-9


# ---- tiers and generation

def test_tier_bands():
    assert tier_for(2.0) == "Win-Win" and tier_for(2.4) == "Win-Win"
    assert tier_for(1.99) == "Worth a Shot" and tier_for(0.5) == "Worth a Shot"
    assert tier_for(0.49) == "Long Shot" and tier_for(LONG_SHOT_MIN) == "Long Shot"
    assert tier_for(-1.51) is None


def test_generate_trades_respects_tiers_and_limits():
    me = core("m", [P("mWR3", "WR", 13), P("mRB3", "RB", 14), P("mRB4", "RB", 12.5)])
    target = core("t", [P("tWR3", "WR", 15), P("tWR4", "WR", 13.5), P("tTE", "TE", 6.0)])
    snap = snapshot(me, target)
    result = generate_trades(snap, 1, 2, min_my_gain=0.0)
    assert set(result) == {"Win-Win", "Worth a Shot", "Long Shot"}
    for tier, proposals in result.items():
        assert len(proposals) <= 2
        for t in proposals:
            assert t.delta_lineup >= 0 and t.mai >= LONG_SHOT_MIN
            assert t.tier == tier and t.why
            if tier == "Worth a Shot":
                assert t.mai >= 0.5 or t.mai >= 2.0  # demoted Win-Wins keep their high MAI
            if tier == "Long Shot":
                assert t.mai < 0.5
    used = [p for ps in result.values() for t in ps for p in t.send]
    assert used or all(not v for v in result.values())


def test_win_win_never_asks_for_an_alpha_asset_without_sending_one():
    fillers = [core(f"f{i}", [P(f"fw{i}", "WR", 10 + i * 0.1)]) for i in range(3)]
    snap = pair([P("a", "WR", 21), P("b", "WR", 20)], [P("Star", "WR", 22)], fillers)
    tc = TradeContext(snap, 1, 2)
    t = tc.evaluate([snap.teams[1].players[-1]], [snap.teams[2].players[-1]])
    assert t.tier != "Win-Win"
