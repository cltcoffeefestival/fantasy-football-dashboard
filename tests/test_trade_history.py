"""Tests for trade history: the learned calibration, how it changes trade scores, the ESPN import
parsing, and the REST calls the store makes."""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trade_finder import Calibration, evaluate_custom  # noqa: E402
from trade_history import (  # noqa: E402
    HistoryStore, TradeRecord, fit_calibration, record_from_proposal, summarize, trade_from_activity,
)
from tests.test_trade_finder import P, core, snapshot  # noqa: E402


def offer(accept, status, multi=False, to="Them"):
    sends = ["A", "B"] if multi else ["A"]
    return TradeRecord(1, 2026, "Me", to, sends, ["X"], status, predicted_accept=accept)


def test_no_answered_offers_leaves_the_engine_untouched():
    cal = fit_calibration([offer(1.0, "sent"), TradeRecord(1, 2026, "A", "B", ["x"], ["y"], "completed", source="espn")])
    assert not cal.trained
    assert cal.adjust(1.3, False) == 1.3


def test_offers_the_model_liked_but_got_rejected_lower_acceptance():
    cal = fit_calibration([offer(1.5, "rejected") for _ in range(6)])
    assert cal.trained and cal.samples == 6
    assert cal.adjust(1.5, False) < 1.5 - 0.5
    assert cal.chance(1.5, False) < Calibration().chance(1.5, False)


def test_a_few_offers_nudge_rather_than_swing():
    one = fit_calibration([offer(1.5, "rejected")])
    many = fit_calibration([offer(1.5, "rejected") for _ in range(20)])
    assert 1.5 - one.adjust(1.5, False) < 1.5 - many.adjust(1.5, False)
    assert 1.5 - one.adjust(1.5, False) < 1.0


def test_rejected_packages_and_accepted_one_for_ones_charge_multi_player_deals_extra():
    records = [offer(1.0, "accepted") for _ in range(5)] + [offer(1.0, "rejected", multi=True) for _ in range(7)]
    cal = fit_calibration(records)
    assert cal.multi < -0.5
    assert cal.adjust(1.0, True) < cal.adjust(1.0, False)


def test_a_counter_counts_as_half_a_yes():
    countered = fit_calibration([offer(0.5, "countered") for _ in range(10)])
    assert abs(countered.adjust(0.5, False) - 0.5) < 0.05


def _hole_filler(calibration=None):
    # the engine's "filling a major lineup hole" Win-Win, plus the same idea as a 2-for-2 package
    me = core("m", [P("mRB3", "RB", 13), P("mRB4", "RB", 12.5)], wr=(14, 8))
    target = core("t", [P("tWR3", "WR", 12.5), P("tWR4", "WR", 11)], rb=(15, 8))
    snap = snapshot(me, target)
    mine, theirs = snap.teams[1].players, snap.teams[2].players
    pick = lambda roster, names: [p for p in roster if p.name in names]  # noqa: E731
    one = evaluate_custom(snap, 1, 2, pick(mine, ["mRB3"]), pick(theirs, ["tWR3"]), calibration=calibration)
    two = evaluate_custom(snap, 1, 2, pick(mine, ["mRB3", "mRB4"]), pick(theirs, ["tWR3", "tWR4"]),
                          calibration=calibration)
    return one, two


def test_the_calibration_moves_tiers_and_keeps_the_raw_score():
    one, _ = _hole_filler()
    assert one.tier == "Win-Win" and one.accept_chance is None and one.learned_shift == 0
    harsh = Calibration(intercept=-3.5, slope=1.0, multi=0.0, samples=10)
    tuned, _ = _hole_filler(harsh)
    assert abs(tuned.raw_acceptance - one.raw_acceptance) < 1e-9
    assert abs(tuned.learned_shift + 3.0) < 1e-9
    assert tuned.tier != "Win-Win" and tuned.accept_chance < Calibration().chance(one.raw_acceptance, False)


def test_a_learned_package_penalty_only_hits_multi_player_deals():
    one, two = _hole_filler()
    hates_packages = Calibration(intercept=-0.5, slope=1.0, multi=-2.0, samples=10)
    one_t, two_t = _hole_filler(hates_packages)
    assert abs(one_t.acceptance - one.acceptance) < 1e-9
    assert abs(two_t.acceptance - (two.acceptance - 2.0)) < 1e-9


def test_a_logged_offer_stores_the_raw_prediction():
    t = SimpleNamespace(target_name="Them", send=[SimpleNamespace(name="A")], receive=[SimpleNamespace(name="X")],
                        raw_acceptance=1.23456, acceptance=0.2, my_effective_gain=2.0, tier="Worth a Shot")
    r = record_from_proposal(t, 1, 2026, "Me", "sent", week=7)
    assert r.predicted_accept == 1.235 and r.to_row()["sends"] == ["A"] and r.week == 7


def test_summary_counts_and_managers():
    records = [offer(1.0, "accepted", to="A"), offer(1.0, "rejected", to="A"), offer(0.0, "sent", to="B"),
               TradeRecord(1, 2026, "A", "C", ["p"], ["q"], "completed", source="espn")]
    s = summarize(records)
    assert (s.answered, s.accepted, s.rejected, s.pending) == (2, 1, 1, 1)
    a = next(m for m in s.by_manager if m["Team"] == "A")
    assert a["Offers sent to"] == 2 and a["Trades completed"] == 1 and a["Accepted"] == 1


def team(name):
    return SimpleNamespace(team_name=name)


def player(name, pid):
    return SimpleNamespace(name=name, playerId=pid)


def test_an_espn_trade_becomes_one_record_with_both_sides():
    a, b = team("Alpha"), team("Beta")
    activity = SimpleNamespace(date=1760000000000, actions=[
        (a, "TRADE_SENT", player("Bijan", 1), 0), (b, "TRADE_RECEIVED", player("Bijan", 1), 0),
        (b, "TRADE_SENT", player("Nacua", 2), 0), (a, "TRADE_RECEIVED", player("Nacua", 2), 0),
        (b, "TRADE_SENT", player("Kittle", 3), 0), (a, "TRADE_RECEIVED", player("Kittle", 3), 0),
    ])
    r = trade_from_activity(activity, 99, 2026)
    assert (r.from_team, r.to_team) == ("Alpha", "Beta")
    assert r.sends == ["Bijan"] and r.receives == ["Nacua", "Kittle"]
    assert r.espn_key == "99:1760000000000:1-2-3" and r.status == "completed" and r.source == "espn"
    assert r.created_at.startswith("2025-10-09")


def test_a_non_trade_activity_is_skipped():
    activity = SimpleNamespace(date=1, actions=[(team("A"), "FA ADDED", player("x", 1), 0)])
    assert trade_from_activity(activity, 1, 2026) is None


class FakeResponse:
    def __init__(self, data=None):
        self.data = data or []

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


class FakeSession:
    def __init__(self, data=None):
        self.calls = []
        self.data = data

    def _call(self, method, url, **kw):
        self.calls.append((method, url, kw))
        return FakeResponse(self.data)

    def get(self, url, **kw):
        return self._call("GET", url, **kw)

    def post(self, url, **kw):
        return self._call("POST", url, **kw)

    def patch(self, url, **kw):
        return self._call("PATCH", url, **kw)


def test_store_is_disabled_without_credentials():
    assert not HistoryStore(url="", key="").enabled


def test_store_requests():
    row = offer(1.0, "sent").to_row() | {"id": 5, "created_at": "2026-10-19T00:00:00Z"}
    s = FakeSession([row])
    store = HistoryStore(url="https://x.supabase.co/", key="sb_secret_abc", session=s)
    assert store.list(1, 2026)[0].id == 5
    method, url, kw = s.calls[-1]
    assert url == "https://x.supabase.co/rest/v1/trade_log" and kw["params"]["league_id"] == "eq.1"
    assert "Authorization" not in kw["headers"] and kw["headers"]["apikey"] == "sb_secret_abc"
    store.import_espn([TradeRecord(1, 2026, "A", "B", ["x"], ["y"], "completed", source="espn", espn_key="k")])
    method, url, kw = s.calls[-1]
    assert kw["params"] == {"on_conflict": "espn_key"} and "ignore-duplicates" in kw["headers"]["Prefer"]
    store.update(5, status="rejected")
    method, url, kw = s.calls[-1]
    assert method == "PATCH" and kw["params"] == {"id": "eq.5"} and kw["json"]["status"] == "rejected"
    jwt = HistoryStore(url="https://x", key="eyJabc", session=FakeSession())
    assert jwt._headers()["Authorization"] == "Bearer eyJabc"
