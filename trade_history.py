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

from trade_finder import PRIOR_PIVOT, Calibration

logger = logging.getLogger(__name__)

TABLE = "trade_log"
STATUSES = ["sent", "accepted", "rejected", "countered"]   # what you can log for your own offer
STATUS_LABELS = {"sent": "Sent, no answer yet", "accepted": "Accepted", "rejected": "Rejected",
                 "countered": "Countered", "completed": "Completed on ESPN"}
# how much of a "yes" each answer is; a counter means interest but not at this price
OUTCOME = {"accepted": 1.0, "countered": 0.5, "rejected": 0.0}
PRIOR_STRENGTH = 3.0         # how many offers' worth of weight the untrained model carries per coefficient
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
        self.key = key if key is not None else os.getenv("SUPABASE_KEY", "")
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


# ------------------------------------------------------------------ learning

def training_rows(records: Iterable[TradeRecord]):
    """(raw acceptance, multi-player, outcome 0..1) for every answered offer with a prediction"""
    return [(r.predicted_accept, r.multi_player, OUTCOME[r.status]) for r in records
            if r.source == "app" and r.status in OUTCOME and r.predicted_accept is not None]


def fit_calibration(records: Iterable[TradeRecord], prior_strength: float = PRIOR_STRENGTH,
                    iterations: int = 50) -> Calibration:
    """Penalized logistic regression of the answer on the engine's prediction, pulled toward the
    untrained model (intercept -PRIOR_PIVOT, slope 1, no multi-player effect)"""
    rows = training_rows(records)
    if not rows:
        return Calibration()
    prior = [-PRIOR_PIVOT, 1.0, 0.0]
    theta = list(prior)
    for _ in range(iterations):
        grad = [prior_strength * (theta[i] - prior[i]) for i in range(3)]
        hess = [[prior_strength if i == j else 0.0 for j in range(3)] for i in range(3)]
        for accept, multi, y in rows:
            x = [1.0, accept, 1.0 if multi else 0.0]
            z = sum(t * v for t, v in zip(theta, x))
            p = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, z))))
            w = p * (1 - p)
            for i in range(3):
                grad[i] += (p - y) * x[i]
                for j in range(3):
                    hess[i][j] += w * x[i] * x[j]
        step = _solve3(hess, grad)
        theta = [t - s for t, s in zip(theta, step)]
        if max(abs(s) for s in step) < 1e-8:
            break
    # a slope at or below zero would mean "the more they gain, the less they accept"; that is noise
    theta[1] = max(theta[1], 0.1)
    return Calibration(intercept=theta[0], slope=theta[1], multi=theta[2], samples=len(rows))


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
    by_manager: List[Dict] = field(default_factory=list)


def summarize(records: List[TradeRecord], calibration: Optional[Calibration] = None) -> LearningSummary:
    """Predicted vs actual, and what the calibration changes, for the History page"""
    cal = calibration or fit_calibration(records)
    mine = [r for r in records if r.source == "app"]
    rows = training_rows(mine)
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
