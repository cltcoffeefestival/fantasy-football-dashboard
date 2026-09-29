"""Regression tests for the lineup math behind the trade finder."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trade_finder import (  # noqa: E402
    LeagueSnapshot, PlayerValue, TeamSnapshot, WAIVER_BASELINES,
    build_slots, cut_order, find_trades, lineup_total, team_analysis, trade_values,
)

SLOTS = build_slots({"QB": 1, "RB": 2, "WR": 2, "TE": 1, "RB/WR/TE": 1, "D/ST": 1, "K": 1, "BE": 6})
WEEKS = 4


def P(name, pos, pts):
    return PlayerValue(name, pos, "X", "ACTIVE", pts, [pts] * WEEKS)


def core(extra):
    return [P("QB", "QB", 20), P("RB1", "RB", 15), P("RB2", "RB", 12), P("WR1", "WR", 14),
            P("WR2", "WR", 12), P("DST", "D/ST", 7), P("K", "K", 7)] + extra


def weekly(players):
    return lineup_total(players, SLOTS, WEEKS) / WEEKS


def without(players, *names):
    return [p for p in players if p.name not in names]


def test_vacant_starter_slot_falls_back_to_waiver_baseline():
    roster = core([P("Bowers", "TE", 12), P("FlexRB", "RB", 9)])
    lost = weekly(roster) - weekly(without(roster, "Bowers"))
    assert abs(lost - (12 - WAIVER_BASELINES["TE"])) < 1e-9


def test_vacant_starter_slot_uses_best_bench_player_when_better_than_waiver():
    roster = core([P("Bowers", "TE", 12), P("BenchTE", "TE", 8.5), P("FlexRB", "RB", 9)])
    lost = weekly(roster) - weekly(without(roster, "Bowers"))
    assert abs(lost - (12 - 8.5)) < 1e-9


def test_opponent_losing_starter_and_bench_player_is_replaced_from_remaining_roster():
    roster = core([P("TE", "TE", 8), P("Watson", "WR", 16.8),
                   P("Montgomery", "RB", 12.7), P("BenchWR", "WR", 10.5)])
    after = without(roster, "Watson", "Montgomery")
    # RB 12.7 -> 12 (-0.7), WR 16.8 -> 12 slot shift (-4.8), flex 12 -> 10.5 (-1.5)
    assert abs((weekly(roster) - weekly(after)) - 7.0) < 1e-9
    # with no bench WR left, the flex falls back to the WR waiver baseline
    thin = without(roster, "BenchWR")
    assert weekly(thin) - weekly(without(thin, "Watson", "Montgomery")) > 0


def test_trade_gains_include_the_lost_starter_on_both_sides():
    me = core([P("Bowers", "TE", 12), P("WRdepth", "WR", 9)])
    other = core([P("TE", "TE", 6.6), P("Watson", "WR", 16.8), P("Montgomery", "RB", 12.7)])
    snap = LeagueSnapshot(
        {1: TeamSnapshot(1, "Me", me), 2: TeamSnapshot(2, "Them", other)},
        SLOTS, list(range(5, 5 + WEEKS)), True, season_weeks_remaining=14,
    )
    trades = find_trades(snap, 1, min_gain=0.0, creative=True)
    assert trades
    for t in trades:
        give, get = {id(p) for p in t.give}, {id(p) for p in t.get}
        my_after = [p for p in me if id(p) not in give] + list(t.get)
        their_after = [p for p in other if id(p) not in get] + list(t.give)
        # gains are measured against full lineups on both sides (drops only apply to uneven swaps)
        if len(t.give) == len(t.get):
            assert abs(t.my_gain - (weekly(my_after) - weekly(me))) < 1e-9
            assert abs(t.their_gain - (weekly(their_after) - weekly(other))) < 1e-9


def test_cut_players_come_from_the_bench_before_starters():
    # A starting K and D/ST sit right at the waiver baseline, so they have zero trade value and a
    # naive "cheapest player" cut would drop them. The cut must come from the bench instead.
    roster = core([P("TE", "TE", 6.6), P("Watson", "WR", 16.8), P("Bench3", "WR", 11), P("Bench4", "RB", 10.5)])
    snap = LeagueSnapshot({1: TeamSnapshot(1, "Them", roster)}, SLOTS, list(range(5, 5 + WEEKS)), True, 14)
    roles = team_analysis(snap, snap.teams[1])
    order = [p.name for p in cut_order(roster, roles, trade_values(snap))]
    starters = {"QB", "RB1", "RB2", "WR1", "WR2", "TE", "DST", "K", "Watson"}
    bench_count = sum(1 for p in roster if roles[id(p)][0] == "Bench")
    assert bench_count >= 1
    assert all(name not in starters for name in order[:bench_count]), order
