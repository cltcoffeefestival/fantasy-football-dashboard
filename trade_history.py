"""
Trade history: offers you log from the app, completed trades pulled from ESPN, and the
calibration the trade engine learns from them.

Storage is one Supabase table (`trade_log`, see `supabase/trade_log.sql`) reached through its REST
API with the project's secret key, so the app keeps working on Streamlit Cloud where local files
vanish on restart. Without SUPABASE_URL and SUPABASE_KEY the store reports itself disabled and the
rest of the app is unchanged.

Learning: every logged offer stores what the engine predicted for the other manager at the time
(the raw acceptance score) next to what actually happened. `fit_calibration` fits a small logistic
model of "did they accept" on that score and on whether the deal was multi-player, pulled toward the
engine's untrained model so a handful of offers nudges it rather than swinging it. The result is a
`trade_finder.Calibration` the Trade Finder applies to every acceptance score.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import math
import os
from typing import Dict, Iterable, List, Optional

import requests

from trade_finder import PRIOR_PIVOT, Calibration, LeagueSnapshot, TeamSnapshot, TradeContext

logger = logging.getLogger(__name__)

TABLE = "trade_log"
STATUSES = ["sent", "accepted", "rejected", "countered"]   # what you can log for your own offer
STATUS_LABELS = {"sent": "Sent, no answer yet", "accepted": "Accepted", "rejected": "Rejected",
                 "countered": "Countered", "completed": "Completed on ESPN"}
# how much of a "yes" each answer is; a counter means interest but not at this price
OUTCOME = {"accepted": 1.0, "countered": 0.5, "rejected": 0.0}
PRIOR_STRENGTH = 3.0         # how many offers' worth of weight the untrained model carries per coefficient
# A completed league trade is a yes from both managers, so each side becomes an accepted example. ESPN
# only shows the trades that happened, never the ones turned down, so a whole trade counts as half an
# offer you logged (which can be a no): a quarter for each side.
COMPLETED_WEIGHT = 0.25
TIMEOUT = 10


# ------------------------------------------------------------------- records

@dataclass
class TradeRecord:
    league_id: int
    season: int
    from_team: str                      # who made the offer (you, for logged offers)
    to_team: str                        # who received it
    sends: List[str]                    # players from_team gives up
    receives: List[str]                 # players from_team gets
    status: str
    source: str = "app"                 # "app" (you logged it) or "espn" (a completed league trade)
    week: Optional[int] = None
    predicted_accept: Optional[float] = None   # the engine's RAW acceptance score when logged
    predicted_mine: Optional[float] = None     # what the engine said the deal was worth to you
    tier: Optional[str] = None
    notes: str = ""
    espn_key: Optional[str] = None
    created_at: Optional[str] = None
    id: Optional[int] = None

    @property
    def multi_player(self) -> bool:
        return not (len(self.sends) == 1 and len(self.receives) == 1)

    def to_row(self) -> Dict:
        row = {
            "league_id": self.league_id, "season": self.season, "week": self.week, "source": self.source,
            "from_team": self.from_team, "to_team": self.to_team, "sends": list(self.sends),
            "receives": list(self.receives), "status": self.status, "predicted_accept": self.predicted_accept,
            "predicted_mine": self.predicted_mine, "tier": self.tier, "notes": self.notes or None,
            "espn_key": self.espn_key,
        }
        if self.created_at:
            row["created_at"] = self.created_at
        return row

    @classmethod
    def from_row(cls, row: Dict) -> "TradeRecord":
        return cls(
            id=row.get("id"), league_id=row["league_id"], season=row["season"], week=row.get("week"),
            source=row.get("source", "app"), from_team=row["from_team"], to_team=row["to_team"],
            sends=list(row.get("sends") or []), receives=list(row.get("receives") or []), status=row["status"],
            predicted_accept=row.get("predicted_accept"), predicted_mine=row.get("predicted_mine"),
            tier=row.get("tier"), notes=row.get("notes") or "", espn_key=row.get("espn_key"),
            created_at=row.get("created_at"),
        )


def record_from_proposal(t, league_id: int, season: int, my_team: str, status: str,
                         week: Optional[int] = None, notes: str = "") -> TradeRecord:
    """A logged offer built from a Trade Finder proposal, keeping the engine's raw prediction"""
    return TradeRecord(
        league_id=league_id, season=season, week=week, from_team=my_team, to_team=t.target_name,
        sends=[p.name for p in t.send], receives=[p.name for p in t.receive], status=status,
        predicted_accept=round(t.raw_acceptance, 3), predicted_mine=round(t.my_effective_gain, 3),
        tier=t.tier, notes=notes,
    )


# ---------------------------------------------------------------- ESPN import

def _player_name(player) -> str:
    return getattr(player, "name", None) or str(player)


def _player_key(player) -> str:
    return str(getattr(player, "playerId", None) or _player_name(player))


def espn_trades(league, size: int = 50) -> List[TradeRecord]:
    """Completed trades in an ESPN league, one record per trade (espn_api's recent_activity)"""
    try:
        activities = league.recent_activity(size=size, msg_type="TRADED")
    except Exception as e:      # private league without cookies, pre-2019 season, ESPN hiccup
        logger.warning(f"Could not read trades for league {getattr(league, 'league_id', '?')}: {e}")
        return []
    records = []
    for activity in activities:
        record = trade_from_activity(activity, league.league_id, league.year)
        if record:
            records.append(record)
    return records


def trade_from_activity(activity, league_id: int, season: int) -> Optional[TradeRecord]:
    """Turn one espn_api Activity into a trade record: who sent what to whom"""
    sent: Dict[str, List] = {}
    order: List[str] = []
    for team, action, player, *_ in activity.actions:
        if action != "TRADE_SENT" or not team:
            continue
        name = getattr(team, "team_name", str(team))
        if name not in sent:
            sent[name] = []
            order.append(name)
        sent[name].append(player)
    if not order:
        return None
    if len(order) == 1:
        # the other side's players can be missing from the feed; find the receiving team instead
        receivers = [getattr(team, "team_name", str(team)) for team, action, *_ in activity.actions
                     if action == "TRADE_RECEIVED" and team]
        other = next((n for n in receivers if n != order[0]), None)
        if other is None:
            return None
        order.append(other)
        sent[other] = []
    a, b = order[0], order[1]
    players = sorted(_player_key(p) for team in order for p in sent[team])
    when = datetime.fromtimestamp(activity.date / 1000, tz=timezone.utc).isoformat() if activity.date else None
    return TradeRecord(
        league_id=league_id, season=season, from_team=a, to_team=b, source="espn", status="completed",
        sends=[_player_name(p) for p in sent[a]], receives=[_player_name(p) for p in sent[b]],
        espn_key=f"{league_id}:{activity.date}:{'-'.join(players)}", created_at=when,
    )


# ------------------------------------------------------------------- storage

class HistoryStore:
    """The trade_log table over Supabase's REST API"""

    def __init__(self, url: Optional[str] = None, key: Optional[str] = None, session=None):
        self.url = (url if url is not None else os.getenv("SUPABASE_URL", "")).rstrip("/")
        # forgive the usual paste slips: spaces, extra quotes, or the <...> from a placeholder
        self.key = (key if key is not None else os.getenv("SUPABASE_KEY", "")).strip().strip("\"'<> ")
        self.session = session or requests.Session()

    @property
    def enabled(self) -> bool:
        return bool(self.url and self.key)

    def _headers(self, prefer: Optional[str] = None) -> Dict[str, str]:
        headers = {"apikey": self.key, "Content-Type": "application/json"}
        if self.key.startswith("eyJ"):       # legacy JWT keys also go in Authorization; new sb_secret_ keys must not
            headers["Authorization"] = f"Bearer {self.key}"
        if prefer:
            headers["Prefer"] = prefer
        return headers

    def _endpoint(self) -> str:
        return f"{self.url}/rest/v1/{TABLE}"

    def list(self, league_id: int, season: int) -> List[TradeRecord]:
        r = self.session.get(
            self._endpoint(), headers=self._headers(), timeout=TIMEOUT,
            params={"league_id": f"eq.{league_id}", "season": f"eq.{season}", "order": "created_at.desc"},
        )
        r.raise_for_status()
        return [TradeRecord.from_row(row) for row in r.json()]

    def add(self, record: TradeRecord) -> TradeRecord:
        r = self.session.post(self._endpoint(), headers=self._headers("return=representation"),
                              json=record.to_row(), timeout=TIMEOUT)
        r.raise_for_status()
        return TradeRecord.from_row(r.json()[0])

    def update(self, record_id: int, **changes) -> None:
        changes["updated_at"] = datetime.now(timezone.utc).isoformat()
        r = self.session.patch(self._endpoint(), headers=self._headers(), json=changes,
                               params={"id": f"eq.{record_id}"}, timeout=TIMEOUT)
        r.raise_for_status()

    def import_espn(self, records: Iterable[TradeRecord]) -> int:
        """Add completed ESPN trades, skipping ones already stored. Returns how many were sent."""
        rows = [r.to_row() for r in records if r.espn_key]
        if not rows:
            return 0
        r = self.session.post(self._endpoint(), headers=self._headers("resolution=ignore-duplicates"),
                              params={"on_conflict": "espn_key"}, json=rows, timeout=TIMEOUT)
        r.raise_for_status()
        return len(rows)


# --------------------------------------------------- completed trades, re-scored

@dataclass
class CompletedScore:
    record: TradeRecord
    from_accept: float          # the engine's raw acceptance for from_team (it said yes)
    to_accept: float            # ... and for to_team (it said yes too)
    multi_player: bool


def _find(players, names):
    """The roster's players with these names, or None if any is missing"""
    found = []
    pool = list(players)
    for name in names:
        match = next((p for p in pool if p.name == name), None)
        if match is None:
            return None
        pool.remove(match)
        found.append(match)
    return found


def score_completed_trade(record: TradeRecord, snapshot: LeagueSnapshot) -> Optional[CompletedScore]:
    """Score a completed trade from both sides by putting the players back where they were.

    Today's rosters hold the trade's result: from_team has what it received, to_team what it sent.
    Swapping them back rebuilds the two rosters as they stood (apart from later moves), and the engine
    then scores the trade as an offer to each side. Uses today's player values, so it is an estimate;
    returns None when a player has since left either roster.
    """
    by_name = {t.name: t for t in snapshot.teams.values()}
    a, b = by_name.get(record.from_team), by_name.get(record.to_team)
    if a is None or b is None or a is b or not (record.sends or record.receives):
        return None
    a_now = _find(a.players, record.receives)       # what A received, now on A
    b_now = _find(b.players, record.sends)          # what A sent, now on B
    if a_now is None or b_now is None:
        return None
    a_ids, b_ids = {id(p) for p in a_now}, {id(p) for p in b_now}
    teams = dict(snapshot.teams)
    teams[a.team_id] = TeamSnapshot(a.team_id, a.name, [p for p in a.players if id(p) not in a_ids] + b_now, a.record)
    teams[b.team_id] = TeamSnapshot(b.team_id, b.name, [p for p in b.players if id(p) not in b_ids] + a_now, b.record)
    before = LeagueSnapshot(teams, snapshot.slots, snapshot.weeks, snapshot.schedule_adjusted,
                            snapshot.drv, snapshot.drv_live, snapshot.drv_sources)
    if not before.weeks:
        return None
    to_side = TradeContext(before, a.team_id, b.team_id).evaluate(b_now, a_now)     # A offers to B
    from_side = TradeContext(before, b.team_id, a.team_id).evaluate(a_now, b_now)   # B offers to A
    return CompletedScore(record, from_side.raw_acceptance, to_side.raw_acceptance, not to_side.simple)


def score_completed_trades(records: Iterable[TradeRecord], snapshot: LeagueSnapshot) -> List[CompletedScore]:
    scores = []
    for r in records:
        if r.source != "espn":
            continue
        try:
            s = score_completed_trade(r, snapshot)
        except Exception as e:      # an odd roster shouldn't take the page down
            logger.warning(f"Could not score trade {r.espn_key}: {e}")
            s = None
        if s:
            scores.append(s)
    return scores


# ------------------------------------------------------------------ learning

def training_rows(records: Iterable[TradeRecord], completed: Iterable[CompletedScore] = ()):
    """(raw acceptance, multi-player, outcome 0..1, weight) for every answered offer with a prediction,
    plus both sides of every re-scored completed trade as a down-weighted yes"""
    rows = [(r.predicted_accept, r.multi_player, OUTCOME[r.status], 1.0) for r in records
            if r.source == "app" and r.status in OUTCOME and r.predicted_accept is not None]
    for c in completed:
        rows.append((c.from_accept, c.multi_player, 1.0, COMPLETED_WEIGHT))
        rows.append((c.to_accept, c.multi_player, 1.0, COMPLETED_WEIGHT))
    return rows


def fit_calibration(records: Iterable[TradeRecord], completed: Iterable[CompletedScore] = (),
                    prior_strength: float = PRIOR_STRENGTH, iterations: int = 50) -> Calibration:
    """Penalized, weighted logistic regression of the answer on the engine's prediction, pulled toward
    the untrained model (intercept -PRIOR_PIVOT, slope 1, no multi-player effect)"""
    records, completed = list(records), list(completed)
    rows = training_rows(records, completed)
    if not rows:
        return Calibration()
    prior = [-PRIOR_PIVOT, 1.0, 0.0]
    theta = list(prior)
    for _ in range(iterations):
        grad = [prior_strength * (theta[i] - prior[i]) for i in range(3)]
        hess = [[prior_strength if i == j else 0.0 for j in range(3)] for i in range(3)]
        for accept, multi, y, weight in rows:
            x = [1.0, accept, 1.0 if multi else 0.0]
            z = sum(t * v for t, v in zip(theta, x))
            p = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, z))))
            w = weight * p * (1 - p)
            for i in range(3):
                grad[i] += weight * (p - y) * x[i]
                for j in range(3):
                    hess[i][j] += w * x[i] * x[j]
        step = _solve3(hess, grad)
        theta = [t - s for t, s in zip(theta, step)]
        if max(abs(s) for s in step) < 1e-8:
            break
    # a slope at or below zero would mean "the more they gain, the less they accept"; that is noise
    theta[1] = max(theta[1], 0.1)
    offers = sum(1 for *_, w in rows if w == 1.0)
    return Calibration(intercept=theta[0], slope=theta[1], multi=theta[2], samples=offers + len(completed))


def _solve3(a, b):
    """Solve a 3x3 linear system (Gaussian elimination with partial pivoting)"""
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(3):
        pivot = max(range(c, 3), key=lambda r: abs(m[r][c]))
        m[c], m[pivot] = m[pivot], m[c]
        for r in range(c + 1, 3):
            f = m[r][c] / m[c][c]
            for k in range(c, 4):
                m[r][k] -= f * m[c][k]
    x = [0.0] * 3
    for r in range(2, -1, -1):
        x[r] = (m[r][3] - sum(m[r][k] * x[k] for k in range(r + 1, 3))) / m[r][r]
    return x


@dataclass
class LearningSummary:
    calibration: Calibration
    answered: int = 0
    accepted: int = 0
    countered: int = 0
    rejected: int = 0
    pending: int = 0
    predicted_rate: float = 0.0              # the untrained model's average chance over answered offers
    actual_rate: float = 0.0                 # share accepted (a counter counts half)
    single_shift: float = 0.0                # change to a 1-for-1's acceptance at the Worth a Shot bar
    multi_shift: float = 0.0                 # the same for a multi-player deal
    completed_used: int = 0                  # completed ESPN trades that could be re-scored
    completed_total: int = 0
    completed_predicted: float = 0.0         # the untrained model's average chance for their sides (all were yes)
    by_manager: List[Dict] = field(default_factory=list)


def summarize(records: List[TradeRecord], completed: Iterable[CompletedScore] = (),
              calibration: Optional[Calibration] = None) -> LearningSummary:
    """Predicted vs actual, and what the calibration changes, for the History page"""
    completed = list(completed)
    cal = calibration or fit_calibration(records, completed)
    mine = [r for r in records if r.source == "app"]
    rows = [row[:3] for row in training_rows(mine)]
    untrained = Calibration()
    s = LearningSummary(calibration=cal)
    s.answered = len(rows)
    s.accepted = sum(1 for r in mine if r.status == "accepted")
    s.countered = sum(1 for r in mine if r.status == "countered")
    s.rejected = sum(1 for r in mine if r.status == "rejected")
    s.pending = sum(1 for r in mine if r.status == "sent")
    if rows:
        s.predicted_rate = sum(untrained.chance(a, m) for a, m, _ in rows) / len(rows)
        s.actual_rate = sum(y for *_, y in rows) / len(rows)
    s.completed_total = sum(1 for r in records if r.source == "espn")
    s.completed_used = len(completed)
    if completed:
        sides = [(c.from_accept, c.multi_player) for c in completed] + [(c.to_accept, c.multi_player) for c in completed]
        s.completed_predicted = sum(untrained.chance(a, m) for a, m in sides) / len(sides)
    if cal.trained:
        s.single_shift = cal.adjust(PRIOR_PIVOT, False) - PRIOR_PIVOT
        s.multi_shift = cal.adjust(PRIOR_PIVOT, True) - PRIOR_PIVOT
    managers: Dict[str, Dict] = {}
    for r in records:
        for name in ((r.from_team, r.to_team) if r.source == "espn" else (r.to_team,)):
            m = managers.setdefault(name, {"Team": name, "Trades completed": 0, "Offers sent to": 0,
                                           "Accepted": 0, "Countered": 0, "Rejected": 0})
            if r.source == "espn":
                m["Trades completed"] += 1
            else:
                m["Offers sent to"] += 1
                if r.status in ("accepted", "countered", "rejected"):
                    m[r.status.title()] += 1
    s.by_manager = sorted(managers.values(), key=lambda m: (-m["Trades completed"], -m["Offers sent to"], m["Team"]))
    return s
