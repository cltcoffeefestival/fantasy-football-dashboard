"""
Trade engine (HATE-G): Human Acceptance Trade Engine & Generator.

Every trade is scored from the TARGET manager's perspective with the Manager Acceptance Index:

    MAI = dLineup - TAP - BCP - TSP - AP

    dLineup  net change in the target's optimal starting lineup (PPG). A slot a rostered player
             can't beat (bye, injury, empty) is filled by the top waiver pickup at that position
             (Dynamic Replacement Value, DRV), so incoming players only count for what they add
             over the starter they displace and lost starters are always charged.
    TAP      Tier-1 Alpha Premium: 20% of the PPG of an S-tier asset the target surrenders.
    BCP      Bench Clutter Penalty: half the value over replacement of each bench player the target
             must cut when it receives more players than it sends.
    TSP      Trade Structure Penalty: 1.5 when the target receives 2+ for 1, 1.0 for a 2-for-2.
    AP       Asymmetry Penalty: 3.0 when the target surrenders 2 starters for 1 starter + 1 bench piece.

Trades are grouped by MAI into three acceptance tiers (Win-Win / Worth a Shot / Long Shot).
"""
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

WEEKS_AHEAD = 4
SEASON_LAST_WEEK = 18

# Slot label -> player positions that may fill it
FLEX_ELIGIBILITY = {
    "RB/WR": {"RB", "WR"},
    "WR/TE": {"WR", "TE"},
    "RB/WR/TE": {"RB", "WR", "TE"},
    "OP": {"QB", "RB", "WR", "TE"},
}
NON_LINEUP_SLOTS = {"BE", "IR", "ER", "TQB", ""}

# Probability a player is available in each upcoming week, by ESPN injury status.
# Indexed by weeks from now; beyond the list the player is treated as healthy.
INJURY_AVAILABILITY = {
    "QUESTIONABLE": [0.9],
    "DOUBTFUL": [0.4, 0.8],
    "OUT": [0.0, 0.3, 0.6, 0.85],
    "INJURY_RESERVE": [0.0, 0.0, 0.2, 0.4, 0.6, 0.8],
    "SUSPENSION": [0.0, 0.4, 0.8],
}

# Matchup: ESPN gives each opponent a rank vs. a position (1..32). We assume 1 is the
# toughest defense and 32 the easiest, and swing a player's value by +/- this fraction.
MATCHUP_SWING = 0.12
MATCHUP_POSITIONS = {"QB": "1", "RB": "2", "WR": "3", "TE": "4"}  # ESPN defaultPositionId

# ---- Manager Acceptance Index
TAP_RATE = 0.20              # of the S-tier asset's PPG
BCP_RATE = 0.5               # of the dropped bench player's value over replacement
TSP_RECEIVES_MORE = 1.5      # target receives more players than it sends (e.g. 2-for-1)
TSP_TWO_FOR_TWO = 1.0
AP_PENALTY = 3.0
# S-tier (Tier-1 Alpha) = top N rostered players at the position across the league
S_TIER_TOP = {"QB": 3, "RB": 5, "WR": 6, "TE": 3}

WIN_WIN_MIN = 2.0
WORTH_A_SHOT_MIN = 0.5
LONG_SHOT_MIN = -1.5
TIERS = ["Win-Win", "Worth a Shot", "Long Shot"]
PER_TIER = 2
POOL_SIZE = 8                # top players per side considered in a swap
WEAK_SLOT_MARGIN = 3.0       # a starter within this many PPG of the waiver pickup is a weak slot
MIN_MY_GAIN = 0.5            # a proposal must be worth sending: at least this much for you (PPG)
STRENGTH_MARGIN = 0.10       # +/-10% vs. league average starters => strength / weakness

# Used only when the free agent lookup fails: rough full-PPR waiver-wire PPG by position
FALLBACK_DRV = {"QB": 13.0, "RB": 8.5, "WR": 9.5, "TE": 6.5, "D/ST": 6.0, "K": 7.0}


@dataclass
class PlayerValue:
    name: str
    position: str
    pro_team: str
    injury: str
    base: float                # blended points per game
    weekly: List[float] = field(default_factory=list)

    @property
    def avg(self) -> float:
        return sum(self.weekly) / len(self.weekly) if self.weekly else 0.0

    @property
    def injured(self) -> bool:
        return self.injury not in ("", "ACTIVE", "NORMAL", "NONE")


@dataclass
class TeamSnapshot:
    team_id: int
    name: str
    players: List[PlayerValue]
    record: str = ""


@dataclass
class LeagueSnapshot:
    teams: Dict[int, TeamSnapshot]
    slots: List[Tuple[str, frozenset]]
    weeks: List[int]
    schedule_adjusted: bool
    drv: Dict[str, float] = field(default_factory=lambda: dict(FALLBACK_DRV))
    drv_live: bool = False     # True when DRV came from the league's actual free agents


@dataclass
class TradeProposal:
    target_id: int
    target_name: str
    send: List[PlayerValue]          # what you send to the target
    receive: List[PlayerValue]       # what the target sends you
    mai: float
    delta_lineup: float              # target's net lineup change (PPG)
    tap: float
    bcp: float
    tsp: float
    ap: float
    my_gain: float                   # your net lineup change (PPG)
    tier: Optional[str]
    target_drops: List[PlayerValue] = field(default_factory=list)
    my_drops: List[PlayerValue] = field(default_factory=list)
    # (incoming player, starter he displaces or None for a waiver-filled slot, PPG gained at the slot)
    upgrades: List[Tuple[PlayerValue, Optional[PlayerValue], float]] = field(default_factory=list)
    alpha_assets: List[PlayerValue] = field(default_factory=list)
    solves_deficiency: bool = False
    notes: List[str] = field(default_factory=list)
    why: str = ""


# ---------------------------------------------------------------- player values

def base_points(avg_points: float, projected_avg: float, total_points: float) -> float:
    """Blend actual and projected per-game scoring; trust actuals more as games pile up"""
    avg, proj = avg_points or 0.0, projected_avg or 0.0
    if avg <= 0 and proj <= 0:
        return 0.0
    if proj <= 0:
        return avg
    if avg <= 0:
        return proj
    games = (total_points or 0.0) / avg
    weight = min(games, 8) / 8 * 0.6
    return weight * avg + (1 - weight) * proj


def availability(injury: str, week_index: int) -> float:
    curve = INJURY_AVAILABILITY.get((injury or "").upper(), [])
    return curve[week_index] if week_index < len(curve) else 1.0


def matchup_factor(rank: Optional[int]) -> float:
    if not rank:
        return 1.0
    return 1 + MATCHUP_SWING * ((rank - 16.5) / 15.5)


def build_player_value(player, weeks: List[int], ratings: Dict[int, Dict[str, Dict[str, int]]]) -> PlayerValue:
    """Turn an espn_api Player into a PlayerValue over the given weeks"""
    injury = (getattr(player, "injuryStatus", "") or "ACTIVE").upper()
    position = getattr(player, "position", "")
    base = base_points(
        getattr(player, "avg_points", 0),
        getattr(player, "projected_avg_points", 0),
        getattr(player, "total_points", 0),
    )
    schedule = getattr(player, "schedule", None) or {}
    stats = getattr(player, "stats", None) or {}

    weekly = []
    for i, week in enumerate(weeks):
        game = schedule.get(str(week)) or schedule.get(week)
        if schedule and not game:
            weekly.append(0.0)  # bye week
            continue
        value = (stats.get(week) or {}).get("projected_points") or base
        if game:
            opp = game.get("team") if isinstance(game, dict) else None
            rank = ratings.get(week, {}).get(position, {}).get(opp)
            value *= matchup_factor(rank)
        weekly.append(value * availability(injury, i))

    return PlayerValue(
        name=player.name,
        position=position,
        pro_team=str(getattr(player, "proTeam", "") or ""),
        injury=injury,
        base=base,
        weekly=weekly,
    )


# -------------------------------------------------------------------- lineups

def build_slots(position_slot_counts: Dict[str, int]) -> List[Tuple[str, frozenset]]:
    """Expand slot counts into a list of (label, eligible positions), most restrictive first"""
    slots = []
    for label, count in position_slot_counts.items():
        if label in NON_LINEUP_SLOTS or not count:
            continue
        eligible = frozenset(FLEX_ELIGIBILITY.get(label, {label}))
        slots.extend([(label, eligible)] * int(count))
    slots.sort(key=lambda s: len(s[1]))
    return slots


def slot_context(slots, drv: Dict[str, float]) -> List[Tuple[frozenset, float]]:
    """Each slot with the waiver pickup that would fill it (best DRV among eligible positions)"""
    return [(eligible, max((drv.get(pos, 0.0) for pos in eligible), default=0.0)) for _, eligible in slots]


def _solve_week(players: List[PlayerValue], ctx, week: int):
    """Best lineup for one week. Returns (points, players who start).

    A slot goes to the best available rostered player, unless a waiver pickup would score more
    (bye, injury, or nobody eligible), in which case it is filled at the waiver value.
    """
    ranked = sorted(players, key=lambda p: p.weekly[week], reverse=True)
    used = set()
    total = 0.0
    starters = []
    for eligible, waiver in ctx:
        for idx, p in enumerate(ranked):
            if idx not in used and p.position in eligible:
                if p.weekly[week] >= waiver:
                    used.add(idx)
                    total += p.weekly[week]
                    starters.append(p)
                else:
                    total += waiver
                break
        else:
            total += waiver
    return total, starters


def lineup_ppg(players: List[PlayerValue], ctx, n_weeks: int) -> float:
    """Average points per week of the optimal lineup over the window"""
    return sum(_solve_week(players, ctx, w)[0] for w in range(n_weeks)) / n_weeks


def starter_ids(players: List[PlayerValue], ctx, n_weeks: int) -> set:
    """Players in the optimal lineup for at least half of the window"""
    counts: Dict[int, int] = {}
    for w in range(n_weeks):
        for p in _solve_week(players, ctx, w)[1]:
            counts[id(p)] = counts.get(id(p), 0) + 1
    return {pid for pid, c in counts.items() if c * 2 >= n_weeks}


def position_strengths(teams: Dict[int, TeamSnapshot], slots) -> Dict[int, Dict[str, float]]:
    """Avg weekly value of each team's starters at each position (dedicated slots only)"""
    counts: Dict[str, int] = {}
    for label, eligible in slots:
        if len(eligible) == 1:
            counts[label] = counts.get(label, 0) + 1
    result = {}
    for tid, team in teams.items():
        result[tid] = {}
        for pos, k in counts.items():
            top = sorted((p.avg for p in team.players if p.position == pos), reverse=True)[:k]
            top += [0.0] * (k - len(top))
            result[tid][pos] = sum(top)
    return result


def needs_table(teams: Dict[int, TeamSnapshot], slots, team_id: int) -> List[Dict]:
    """Per-position strength of one team vs. the league average"""
    strengths = position_strengths(teams, slots)
    rows = []
    for pos in strengths[team_id]:
        league_avg = sum(s[pos] for s in strengths.values()) / len(strengths)
        mine = strengths[team_id][pos]
        diff = (mine - league_avg) / league_avg if league_avg else 0.0
        label = "Strength" if diff >= STRENGTH_MARGIN else "Weakness" if diff <= -STRENGTH_MARGIN else "Average"
        rows.append({
            "Position": pos,
            "Starters (pts/wk)": round(mine, 1),
            "League avg": round(league_avg, 1),
            "vs League": f"{diff:+.0%}",
            "Status": label,
        })
    return rows


# ------------------------------------------------------------------------ MAI

def value_over_replacement(player: PlayerValue, drv: Dict[str, float]) -> float:
    """V_p = max(0, projected PPG - top waiver PPG at the position)"""
    return max(0.0, player.base - drv.get(player.position, 0.0))


def s_tier_ids(snapshot: LeagueSnapshot) -> set:
    """Ids of the league's Tier-1 Alpha assets: the top few rostered players at each position"""
    ids = set()
    for pos, top in S_TIER_TOP.items():
        ranked = sorted(
            (p for t in snapshot.teams.values() for p in t.players if p.position == pos),
            key=lambda p: p.base,
            reverse=True,
        )
        ids.update(id(p) for p in ranked[:top])
    return ids


def tier_for(mai: float) -> Optional[str]:
    if mai >= WIN_WIN_MIN:
        return "Win-Win"
    if mai >= WORTH_A_SHOT_MIN:
        return "Worth a Shot"
    if mai >= LONG_SHOT_MIN:
        return "Long Shot"
    return None


class TradeContext:
    """Everything about one (you, target) pair that doesn't change between candidate trades"""

    def __init__(self, snapshot: LeagueSnapshot, my_id: int, target_id: int):
        self.snapshot = snapshot
        self.me = snapshot.teams[my_id]
        self.target = snapshot.teams[target_id]
        self.n = len(snapshot.weeks)
        self.drv = snapshot.drv
        self.ctx = slot_context(snapshot.slots, snapshot.drv)
        self.s_tier = s_tier_ids(snapshot)

        self.my_before = lineup_ppg(self.me.players, self.ctx, self.n)
        self.target_before = lineup_ppg(self.target.players, self.ctx, self.n)
        self.my_starters = starter_ids(self.me.players, self.ctx, self.n)
        self.target_starters = starter_ids(self.target.players, self.ctx, self.n)
        self.my_cut = self._cut_order(self.me.players, self.my_starters)
        self.target_cut = self._cut_order(self.target.players, self.target_starters)

        strengths = needs_table(snapshot.teams, snapshot.slots, target_id)
        self.weak_positions = {r["Position"] for r in strengths if r["Status"] == "Weakness"}

    def _cut_order(self, players, starters):
        """Cheapest to cut first: bench before starters, then lowest value over replacement"""
        return sorted(
            players,
            key=lambda p: (id(p) in starters, value_over_replacement(p, self.drv), p.base),
        )

    def evaluate(self, send: List[PlayerValue], receive: List[PlayerValue]) -> TradeProposal:
        """Score `send` (from you) for `receive` (from the target) from the target's perspective"""
        send_ids = {id(p) for p in send}
        recv_ids = {id(p) for p in receive}

        # whoever ends up with extra players must cut their cheapest bench players
        target_drops = [p for p in self.target_cut if id(p) not in recv_ids][: max(0, len(send) - len(receive))]
        my_drops = [p for p in self.my_cut if id(p) not in send_ids][: max(0, len(receive) - len(send))]
        target_drop_ids = {id(p) for p in target_drops}
        my_drop_ids = {id(p) for p in my_drops}

        target_after = [
            p for p in self.target.players if id(p) not in recv_ids and id(p) not in target_drop_ids
        ] + list(send)
        my_after = [
            p for p in self.me.players if id(p) not in send_ids and id(p) not in my_drop_ids
        ] + list(receive)

        delta_lineup = lineup_ppg(target_after, self.ctx, self.n) - self.target_before
        my_gain = lineup_ppg(my_after, self.ctx, self.n) - self.my_before

        alpha = [p for p in receive if id(p) in self.s_tier]
        tap = TAP_RATE * max(p.base for p in alpha) if alpha else 0.0
        bcp = BCP_RATE * sum(value_over_replacement(p, self.drv) for p in target_drops)
        if len(send) > len(receive):
            tsp = TSP_RECEIVES_MORE
        elif len(send) == len(receive) == 2:
            tsp = TSP_TWO_FOR_TWO
        else:
            tsp = 0.0

        post_starters = starter_ids(target_after, self.ctx, self.n)
        incoming_starters = [p for p in send if id(p) in post_starters]
        surrendered_starters = [p for p in receive if id(p) in self.target_starters]
        ap = 0.0
        if len(surrendered_starters) == 2 and len(send) == 2 and len(incoming_starters) == 1:
            ap = AP_PENALTY

        upgrades = self._upgrades(incoming_starters, recv_ids, post_starters)
        weak = any(
            displaced is None or displaced.base - self.drv.get(displaced.position, 0.0) <= WEAK_SLOT_MARGIN
            for _, displaced, _ in upgrades
        ) or any(p.position in self.weak_positions for p in incoming_starters)

        mai = delta_lineup - tap - bcp - tsp - ap
        return TradeProposal(
            target_id=self.target.team_id,
            target_name=self.target.name,
            send=list(send),
            receive=list(receive),
            mai=mai,
            delta_lineup=delta_lineup,
            tap=tap, bcp=bcp, tsp=tsp, ap=ap,
            my_gain=my_gain,
            tier=self._assign_tier(mai, send, receive, alpha, weak),
            target_drops=target_drops,
            my_drops=my_drops,
            upgrades=upgrades,
            alpha_assets=alpha,
            solves_deficiency=weak,
            notes=_notes(send, receive),
        )

    def _upgrades(self, incoming_starters, recv_ids, post_starters):
        """Which of the target's starters each incoming starter displaces (None = waiver-filled slot)"""
        displaced_pool = [
            p for p in self.target.players
            if id(p) in self.target_starters and id(p) not in recv_ids and id(p) not in post_starters
        ]
        displaced_pool.sort(key=lambda p: p.base)
        upgrades = []
        for p in sorted(incoming_starters, key=lambda p: p.base, reverse=True):
            match = next((d for d in displaced_pool if d.position == p.position), None) or (
                displaced_pool[0] if displaced_pool else None
            )
            if match is not None:
                displaced_pool.remove(match)
                gain = max(0.0, p.base - match.base)
            else:
                gain = max(0.0, p.base - self.drv.get(p.position, 0.0))
            upgrades.append((p, match, gain))
        return upgrades

    def _assign_tier(self, mai, send, receive, alpha, solves_deficiency) -> Optional[str]:
        tier = tier_for(mai)
        if tier == "Win-Win":
            simple = (len(send), len(receive)) in ((1, 1), (2, 2))
            unearned_alpha = bool(alpha) and not any(id(p) in self.s_tier for p in send)
            if not (simple and solves_deficiency and not unearned_alpha):
                tier = "Worth a Shot"
        return tier


def _notes(send, receive) -> List[str]:
    notes = []
    for p in list(send) + list(receive):
        if p.injured:
            notes.append(f"{p.name} is {p.injury.replace('_', ' ').title()}")
        elif p.weekly and p.weekly.count(0.0):
            notes.append(f"{p.name} has a bye in the window")
    return notes


# ------------------------------------------------------------------ generation

def generate_trades(
    snapshot: LeagueSnapshot,
    my_id: int,
    target_id: int,
    min_my_gain: float = MIN_MY_GAIN,
    per_tier: int = PER_TIER,
) -> Dict[str, List[TradeProposal]]:
    """1 to `per_tier` proposals in each acceptance tier for one target team.

    A proposal must give the target a non-negative lineup change and give you at least
    `min_my_gain`; within a tier the ones that help you most come first.
    """
    result: Dict[str, List[TradeProposal]] = {t: [] for t in TIERS}
    if my_id not in snapshot.teams or target_id not in snapshot.teams or my_id == target_id:
        return result

    tc = TradeContext(snapshot, my_id, target_id)
    if tc.n == 0:
        return result

    def pool(players):
        return sorted((p for p in players if p.base > 0), key=lambda p: p.base, reverse=True)[:POOL_SIZE]

    my_pool, their_pool = pool(tc.me.players), pool(tc.target.players)
    candidates: Dict[str, List[TradeProposal]] = {t: [] for t in TIERS}
    for send_n, recv_n in [(1, 1), (2, 1), (1, 2), (2, 2)]:
        for send in combinations(my_pool, send_n):
            for receive in combinations(their_pool, recv_n):
                t = tc.evaluate(list(send), list(receive))
                if t.tier is None or t.delta_lineup < 0 or t.my_gain < min_my_gain:
                    continue
                candidates[t.tier].append(t)

    for tier, found in candidates.items():
        found.sort(key=lambda t: (-t.my_gain, -t.mai))
        used_send, used_recv = set(), set()
        for t in found:
            s_ids, r_ids = {id(p) for p in t.send}, {id(p) for p in t.receive}
            if s_ids & used_send or r_ids & used_recv:
                continue
            t.why = explain(t)
            result[tier].append(t)
            used_send |= s_ids
            used_recv |= r_ids
            if len(result[tier]) >= per_tier:
                break
    return result


def evaluate_custom(snapshot: LeagueSnapshot, my_id: int, target_id: int, send, receive) -> TradeProposal:
    """Score a specific trade you typed in (same MAI as the generated ones)"""
    tc = TradeContext(snapshot, my_id, target_id)
    t = tc.evaluate(list(send), list(receive))
    t.why = explain(t)
    return t


# ----------------------------------------------------------------- explanations

def _names(players) -> str:
    return " + ".join(p.name for p in players)


def _structure(t: TradeProposal) -> str:
    return f"{len(t.send)}-for-{len(t.receive)}"


def _upgrade_text(t: TradeProposal) -> str:
    parts = []
    for p, displaced, gain in t.upgrades:
        over = (
            f"over {displaced.name} ({displaced.base:.1f})"
            if displaced else "in a slot they'd otherwise fill from waivers"
        )
        parts.append(f"{p.name} ({p.base:.1f} PPG) would start at {p.position} {over}, +{gain:.1f} PPG")
    return "; ".join(parts)


def _frictions(t: TradeProposal) -> List[str]:
    out = []
    if t.tsp:
        out.append(f"{_structure(t)} structure (-{t.tsp:.1f})")
    if t.bcp:
        out.append(f"they'd have to cut {_names(t.target_drops)} (-{t.bcp:.1f})")
    if t.tap:
        out.append(f"they'd be giving up S-tier {_names(t.alpha_assets)} (-{t.tap:.1f} Alpha tax)")
    if t.ap:
        out.append(f"two starters out for one starter plus a bench piece (-{t.ap:.1f})")
    return out


def explain(t: TradeProposal) -> str:
    """One paragraph in the voice the tier calls for"""
    upside = _upgrade_text(t)
    friction = _frictions(t)
    if t.tier == "Win-Win":
        return (
            f"Fills a weak starting slot for {t.target_name}: {upside or 'improves their lineup'}. "
            f"Clean {_structure(t)} with no Alpha asset requested, so there's little friction."
        )
    if t.tier == "Worth a Shot":
        hesitate = "; ".join(friction) if friction else (
            "the gain is real but modest, near the point where owner bias decides it"
        )
        return f"Upside for them: {upside or f'+{t.delta_lineup:.1f} PPG in their lineup'}. Hesitation: {hesitate}."
    reasons = "; ".join(friction) if friction else "the lineup gain for them is marginal"
    pitch = []
    for p in t.alpha_assets:
        if p.injured:
            pitch.append(f"{p.name} is {p.injury.replace('_', ' ').title()}, which is the opening to ask for him")
        elif p.weekly and p.weekly.count(0.0):
            pitch.append(f"{p.name} is on a bye inside the window, so his short-term value to them is lower")
    if t.upgrades:
        p, _, gain = t.upgrades[0]
        pitch.append(
            f"lead with the {p.position} upgrade: +{gain:.1f} PPG at the slot, "
            f"+{t.delta_lineup:.1f} PPG for their lineup"
        )
    if t.target_drops:
        pitch.append("offer to take the roster clutter off their hands")
    return (
        f"They'll likely resist: {reasons}. "
        f"How to pitch it: {'; '.join(pitch) or 'frame it around the lineup upgrade for them'}."
    )


# ------------------------------------------------------------ ESPN integration

def _fetch_ratings(league, weeks: List[int]) -> Dict[int, Dict[str, Dict[str, int]]]:
    """{week: {position: {opposing pro team abbrev: rank}}}; empty for weeks that fail"""
    from espn_api.football.constant import PRO_TEAM_MAP

    abbrev = {str(k): v for k, v in PRO_TEAM_MAP.items()}
    pos_by_id = {v: k for k, v in MATCHUP_POSITIONS.items()}
    out: Dict[int, Dict[str, Dict[str, int]]] = {}
    for week in weeks:
        try:
            raw = league._get_positional_ratings(week)
        except Exception as e:
            logger.warning(f"Could not load matchup ratings for week {week}: {e}")
            continue
        out[week] = {
            pos_by_id[pid]: {abbrev.get(tid, tid): rank for tid, rank in by_team.items()}
            for pid, by_team in raw.items()
            if pid in pos_by_id
        }
    return out


def fetch_drv(league, week: int) -> Tuple[Dict[str, float], bool]:
    """Top waiver PPG at each position from the league's actual free agents.

    Returns (drv, live). Positions whose lookup fails fall back to FALLBACK_DRV, and live is
    False unless every position came from real free agents.
    """
    drv = dict(FALLBACK_DRV)
    live = True
    for pos in FALLBACK_DRV:
        try:
            best = 0.0
            for fa in league.free_agents(week=week, size=25, position=pos):
                status = (getattr(fa, "injuryStatus", "") or "ACTIVE").upper()
                if status in ("OUT", "INJURY_RESERVE", "SUSPENSION"):
                    continue
                best = max(best, base_points(
                    getattr(fa, "avg_points", 0),
                    getattr(fa, "projected_avg_points", 0),
                    getattr(fa, "total_points", 0),
                ))
            if best > 0:
                drv[pos] = best
            else:
                live = False
        except Exception as e:
            logger.warning(f"Could not load free agents for {pos}: {e}")
            live = False
    return drv, live


def build_league_snapshot(league, weeks_ahead: int = WEEKS_AHEAD) -> LeagueSnapshot:
    """Read an espn_api League into plain data the engine can work on"""
    current = getattr(league, "current_week", None) or getattr(league, "nfl_week", 1)
    weeks = list(range(current, min(current + weeks_ahead, SEASON_LAST_WEEK + 1)))
    ratings = _fetch_ratings(league, weeks)
    drv, drv_live = fetch_drv(league, current)

    teams = {}
    for team in league.teams:
        teams[team.team_id] = TeamSnapshot(
            team_id=team.team_id,
            name=team.team_name,
            players=[build_player_value(p, weeks, ratings) for p in team.roster],
            record=f"{team.wins}-{team.losses}",
        )

    return LeagueSnapshot(
        teams=teams,
        slots=build_slots(league.settings.position_slot_counts),
        weeks=weeks,
        schedule_adjusted=bool(ratings),
        drv=drv,
        drv_live=drv_live,
    )
