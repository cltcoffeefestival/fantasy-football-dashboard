"""
Trade engine (M-HATE): Mutual Need & Human Acceptance Trade Engine.

A trade is only surfaced when it works for BOTH managers.

Your side       dLineup(you) = net change in your optimal starting lineup (PPG). Incoming players count
                only for what they add over the starter they displace; a redundant second QB/TE
                (when you already have a top-tier QB1/TE1 in a 1QB/1TE format) counts for 0.
Target side     MAI = dLineup(target) - TAP - BCP - TSP - PLT
    TAP  Tier-1 Alpha Premium: 20% of the PPG of an S-tier asset the target surrenders.
    BCP  Bench Clutter Penalty: half the value over replacement of each bench player the target cuts
         when it receives 2+ players.
    TSP  Trade Structure Penalty: 1-for-1 0, 2-for-2 1.0, target receives 2 for 1 -> 1.5,
         target receives 1 for 2 -> 2.5.
    PLT  Positional Loss Tax: the target's starting QB1/TE1 (1QB/1TE formats) or any premier S-tier
         starter, minus the waiver pickup, when it surrenders him without a starter back at the position.
    AP   Asymmetry Penalty: the target surrenders 2+ starters for fewer starters plus a bench piece.
    LAT  Lateral tax: same-position swaps. Also, QB upgrades count half in the target's lineup change.
    TFL  Talent floor: the best player the target gets must be close to the best it gives up.
Mutual       NMU = dLineup(you) + MAI. Tiers require both sides to clear a bar.

A slot no rostered player can fill (or beat) is filled by the top waiver pickup at that position
(Dynamic Replacement Value, DRV), so lost starters are always charged. Each roster is only ever
shortened by cutting the cheapest bench players, so roster limits are respected on both sides.
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

# ---- Target acceptance (MAI)
TAP_RATE = 0.20              # of the S-tier asset's PPG
BCP_RATE = 0.5               # of the dropped bench player's value over replacement
# Calibrated on real outcomes: every multi-player offer we tried was rejected (7 of 7), while the
# simple 1-for-1s were entertained. Structure friction is therefore much heavier than a bare +/-1.
TSP_TWO_FOR_TWO = 2.0
TSP_TARGET_RECEIVES_MORE = 3.0    # e.g. target receives 2, gives 1
TSP_TARGET_RECEIVES_FEWER = 3.5   # e.g. target receives 1, gives 2
AP_PENALTY = 3.0                  # target surrenders 2+ starters for fewer starters plus a bench piece
TAP_PACKAGE_MULT = 2.0            # Alpha tax doubles when it's a depth package for the star
VOID_RATE = 0.5                   # surrendering a premier (S-tier) starter with no starter back at the position
LATERAL_TAX = 1.5                 # same-position swaps give the target no reason to bother
PLT_QB_TE_RATE = 0.5              # lost QB1/TE1: the lineup change already charges the loss, so this is extra
TALENT_FLOOR_RATIO = 0.85         # the best player the target gets should be within 15% of its best player out
TALENT_GAP_RATE = 0.5             # PPG penalty per PPG of talent gap beyond that
SELL_LOW_RATE = 0.5               # your side: PPG deducted per PPG of talent you give beyond what you get back
TALENT_GAP_HOLE_RATE = 0.25       # gentler when a different position's real hole is being filled
REAL_MARGIN = 1.0                 # a player must beat the waiver pickup by this much to count as a real piece
DRV_TOP_N = 3                     # replacement level = average of the best few free agents, not the single best
NON_TRADE_POSITIONS = {"K", "D/ST"}   # nobody trades for these; keep them out of the candidate pools
QB_GAIN_HAIRCUT = 0.5             # owners discount QB upgrades in 1QB leagues: waiver QBs are plentiful
# S-tier (Tier-1 Alpha) = top N rostered players at the position across the league
S_TIER_TOP = {"QB": 3, "RB": 5, "WR": 6, "TE": 3}
STRICT_MIN_MAI = 1.5         # the "strict acceptance" filter: only trades the target clearly clears

# ---- Tiers: (min target MAI, max target MAI (exclusive, None = open), min your dLineup)
WIN_WIN = ("Win-Win", 2.0, None, 1.5)
WORTH_A_SHOT = ("Worth a Shot", 0.5, 2.0, 1.0)
LONG_SHOT = ("Long Shot", -1.5, 0.5, 3.0)
TIERS = ["Win-Win", "Worth a Shot", "Long Shot"]
PER_TIER = 3
POOL_SIZE = 8                # top players per side considered in a swap
WEAK_SLOT_MARGIN = 3.0       # a starter within this many PPG of the waiver pickup is a weak slot
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
    drv_sources: Dict[str, List[Tuple[str, float, float, float]]] = field(default_factory=dict)
    # position -> the free agents behind the waiver level: (name, blended PPG, season avg, projection)


@dataclass
class SideImpact:
    """What a trade does to one team's starting lineup"""
    delta: float = 0.0                                   # net starting PPG change
    perceived: float = 0.0                               # delta as an owner sees it (QB upgrades discounted)
    # (incoming starter, starter he displaces or None for a waiver-filled slot, PPG gained at the slot)
    upgrades: List[Tuple[PlayerValue, Optional[PlayerValue], float]] = field(default_factory=list)
    drops: List[PlayerValue] = field(default_factory=list)       # bench players cut to make room
    lost_starters: List[PlayerValue] = field(default_factory=list)
    redundant: List[PlayerValue] = field(default_factory=list)   # incoming players that add nothing
    after_roster: List[PlayerValue] = field(default_factory=list)   # lineup-eligible roster after the trade
    lineup_before: List[Tuple[str, str, float]] = field(default_factory=list)
    lineup_after: List[Tuple[str, str, float]] = field(default_factory=list)
    slot_notes: Dict[int, str] = field(default_factory=dict)     # id(incoming) -> why a slot counts as waiver-filled
    solves_hole: bool = False                                    # fills a weak slot / weak position
    blocked: bool = False                                        # gives up starters for a redundant bench piece


@dataclass
class TradeProposal:
    target_id: int
    target_name: str
    send: List[PlayerValue]          # what you send to the target
    receive: List[PlayerValue]       # what the target sends you
    mine: SideImpact
    theirs: SideImpact
    tap: float
    bcp: float
    tsp: float
    plt: float
    ap: float = 0.0
    lat: float = 0.0
    tfl: float = 0.0
    sell_low: float = 0.0            # you'd be selling a much better player for a lesser one
    real_in_players: List[PlayerValue] = field(default_factory=list)   # players the target receives who would start for it
    real_in: int = 0                 # ... and how many
    real_out: int = 0                # players the target gives who are starters or beat one
    alpha_assets: List[PlayerValue] = field(default_factory=list)
    tier: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    you_why: str = ""
    them_why: str = ""

    @property
    def mai(self) -> float:
        """Manager Acceptance Index: how likely the target is to say yes"""
        return self.theirs.perceived - self.tap - self.bcp - self.tsp - self.plt - self.ap - self.lat - self.tfl

    @property
    def my_gain(self) -> float:
        """Your raw lineup change (PPG)"""
        return self.mine.delta

    @property
    def my_effective_gain(self) -> float:
        """Your lineup change after the sell-low penalty: what the trade is really worth to you"""
        return self.mine.delta - self.sell_low

    @property
    def delta_lineup(self) -> float:
        """The target's net lineup change"""
        return self.theirs.delta

    @property
    def nmu(self) -> float:
        """Net Mutual Utility: your lineup change plus the target's MAI"""
        return self.my_effective_gain + self.mai

    @property
    def blocked(self) -> bool:
        return self.mine.blocked or self.theirs.blocked

    @property
    def penalties(self) -> List[Tuple[str, float]]:
        """Friction penalties that actually apply"""
        named = [("TAP", self.tap), ("BCP", self.bcp), ("TSP", self.tsp), ("PLT", self.plt),
                 ("AP", self.ap), ("LAT", self.lat), ("TFL", self.tfl)]
        return [(n, v) for n, v in named if v]


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


def single_slot_positions(slots) -> set:
    """Positions in a 1QB / 1TE format.

    QB is single when there is exactly one QB slot and no superflex (OP) slot that could hold a second.
    TE is single when there is exactly one dedicated TE slot; a flex spot doesn't change that.
    """
    result = set()
    dedicated = {pos: sum(1 for _, elig in slots if elig == frozenset({pos})) for pos in ("QB", "TE")}
    if dedicated["QB"] == 1 and not any("QB" in elig and len(elig) > 1 for _, elig in slots):
        result.add("QB")
    if dedicated["TE"] == 1:
        result.add("TE")
    return result


def _solve_week(players: List[PlayerValue], ctx, week: int):
    """Best lineup for one week. Returns (points, players who start, per-slot assignments).

    A slot goes to the best available rostered player, unless a waiver pickup would score more
    (bye, injury, or nobody eligible), in which case it is filled at the waiver value.
    """
    ranked = sorted(players, key=lambda p: p.weekly[week], reverse=True)
    used = set()
    total = 0.0
    starters = []
    assigned = []   # per slot: (player or None for a waiver pickup, points)
    for eligible, waiver in ctx:
        for idx, p in enumerate(ranked):
            if idx not in used and p.position in eligible:
                if p.weekly[week] >= waiver:
                    used.add(idx)
                    total += p.weekly[week]
                    starters.append(p)
                    assigned.append((p, p.weekly[week]))
                else:
                    total += waiver
                    assigned.append((None, waiver))
                break
        else:
            total += waiver
            assigned.append((None, waiver))
    return total, starters, assigned


def lineup_rows(players: List[PlayerValue], slots, ctx, n_weeks: int) -> List[Tuple[str, str, float]]:
    """(slot, who usually fills it, PPG) for the optimal lineup, averaged over the window"""
    per_slot: List[List[Tuple[Optional[PlayerValue], float]]] = [[] for _ in ctx]
    for w in range(n_weeks):
        for i, entry in enumerate(_solve_week(players, ctx, w)[2]):
            per_slot[i].append(entry)
    rows = []
    for (label, _), entries in zip(slots, per_slot):
        names: Dict[str, int] = {}
        for p, _ in entries:
            key = p.name if p else "Waiver pickup"
            names[key] = names.get(key, 0) + 1
        who = max(names, key=names.get)
        rows.append((label, who, sum(pts for _, pts in entries) / n_weeks))
    return rows


def lineup_ppg(players: List[PlayerValue], ctx, n_weeks: int) -> float:
    """Average points per week of the optimal lineup over the window"""
    return sum(_solve_week(players, ctx, w)[0] for w in range(n_weeks)) / n_weeks


def start_counts(players: List[PlayerValue], ctx, n_weeks: int) -> Dict[int, int]:
    """How many weeks of the window each player is in the optimal lineup"""
    counts: Dict[int, int] = {}
    for w in range(n_weeks):
        for p in _solve_week(players, ctx, w)[1]:
            counts[id(p)] = counts.get(id(p), 0) + 1
    return counts


def starter_ids(players: List[PlayerValue], ctx, n_weeks: int) -> set:
    """Players in the optimal lineup for at least half of the window"""
    return {pid for pid, c in start_counts(players, ctx, n_weeks).items() if c * 2 >= n_weeks}


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


def structure_penalty(received: int, sent: int) -> float:
    """TSP from the target's view: it receives `received` real players and sends `sent`.

    Callers pass EFFECTIVE counts: a player who doesn't beat a waiver pickup isn't a real piece.
    """
    if sent == 0:
        return 0.0
    if received == sent:
        return 0.0 if received <= 1 else TSP_TWO_FOR_TWO
    return TSP_TARGET_RECEIVES_MORE if received > sent else TSP_TARGET_RECEIVES_FEWER


def single_slot_ppg(players: List[PlayerValue], pos: str, drv: Dict[str, float], n_weeks: int) -> float:
    """PPG of the lone starter at a one-slot position: best rostered player or the waiver pickup"""
    total = 0.0
    for w in range(n_weeks):
        best = max((p.weekly[w] for p in players if p.position == pos), default=0.0)
        total += max(best, drv.get(pos, 0.0))
    return total / n_weeks


class Side:
    """One team's pre-trade lineup facts"""

    def __init__(self, team: TeamSnapshot, tc: "TradeContext"):
        self.team = team
        self.qb_before = single_slot_ppg(team.players, "QB", tc.drv, tc.n) if "QB" in tc.single else 0.0
        self.before = lineup_ppg(team.players, tc.ctx, tc.n)
        self.starters = starter_ids(team.players, tc.ctx, tc.n)
        # cheapest to cut first: bench before starters, then lowest value over replacement
        self.cut = sorted(
            team.players,
            key=lambda p: (id(p) in self.starters, value_over_replacement(p, tc.drv), p.base),
        )


class TradeContext:
    """Everything about one (you, target) pair that doesn't change between candidate trades"""

    def __init__(self, snapshot: LeagueSnapshot, my_id: int, target_id: int, min_gain: float = WORTH_A_SHOT[3]):
        self.snapshot = snapshot
        self.min_gain = min_gain     # your smallest gain for a Worth a Shot trade (PPG)
        self.n = len(snapshot.weeks)
        self.drv = snapshot.drv
        self.ctx = slot_context(snapshot.slots, snapshot.drv)
        self.s_tier = s_tier_ids(snapshot)
        self.single = single_slot_positions(snapshot.slots)
        self.mine = Side(snapshot.teams[my_id], self)
        self.theirs = Side(snapshot.teams[target_id], self)

        def weak(team_id):
            return {r["Position"] for r in needs_table(snapshot.teams, snapshot.slots, team_id) if r["Status"] == "Weakness"}

        self.my_weak, self.their_weak = weak(my_id), weak(target_id)

    # ---- one side of a trade
    def _side_impact(self, side: Side, weak_positions: set, outgoing, incoming, whose: str = "their") -> SideImpact:
        out_ids = {id(p) for p in outgoing}
        drops = [p for p in side.cut if id(p) not in out_ids][: max(0, len(incoming) - len(outgoing))]
        drop_ids = {id(p) for p in drops}

        # elite starters who stay: incoming players at their position are redundant
        staying_elite = {
            p.position for p in side.team.players
            if id(p) in side.starters and id(p) in self.s_tier and id(p) not in out_ids
        }
        # Incoming players that add nothing an owner could not get from waivers contribute 0:
        #  - 1QB / 1TE: a second QB/TE behind a top-tier QB1/TE1
        #  - anyone who doesn't clearly beat the waiver pickup at his position
        excluded_ids = {
            id(p) for p in incoming
            if (p.position in self.single and p.position in staying_elite)
            or value_over_replacement(p, self.drv) < REAL_MARGIN
        }
        base_roster = [p for p in side.team.players if id(p) not in out_ids and id(p) not in drop_ids]
        roster = base_roster + [p for p in incoming if id(p) not in excluded_ids]
        # a backup who only starts to cover a bye or an injury adds nothing either: a waiver
        # pickup does the same job. Only players who start nearly every week count.
        full_time = max(1, self.n - 1)
        counts = start_counts(roster, self.ctx, self.n)
        part_time = {id(p) for p in incoming if id(p) not in excluded_ids and counts.get(id(p), 0) < full_time}
        if part_time:
            excluded_ids |= part_time
            roster = base_roster + [p for p in incoming if id(p) not in excluded_ids]
        # excluded players are still on the roster (bench), they just can't start
        delta = lineup_ppg(roster, self.ctx, self.n) - side.before
        perceived = delta
        if "QB" in self.single:
            qb_gain = max(0.0, single_slot_ppg(roster, "QB", self.drv, self.n) - side.qb_before)
            perceived = delta - QB_GAIN_HAIRCUT * qb_gain

        post_starters = starter_ids(roster, self.ctx, self.n)
        incoming_starters = [p for p in incoming if id(p) in post_starters]
        lost = [p for p in outgoing if id(p) in side.starters]

        redundant = [p for p in incoming if id(p) not in post_starters and p.position in staying_elite]
        upgrades = self._upgrades(side, incoming_starters, out_ids, post_starters)
        solves = any(
            displaced is None or displaced.base - self.drv.get(displaced.position, 0.0) <= WEAK_SLOT_MARGIN
            for _, displaced, _ in upgrades
        ) or any(p.position in weak_positions for p in incoming_starters)

        slot_notes = {}
        for p, displaced, _ in upgrades:
            if displaced is None:
                best = max(
                    (q for q in side.team.players if q.position == p.position and id(q) not in out_ids),
                    key=lambda q: q.base, default=None,
                )
                waiver = self.drv.get(p.position, 0.0)
                slot_notes[id(p)] = (
                    f"the model fills that slot from waivers ({waiver:.1f} PPG); {whose} best rostered "
                    f"{p.position} is {best.name} ({best.base:.1f})" if best is not None
                    else f"{whose} roster has no {p.position}, so the model fills it from waivers ({waiver:.1f} PPG)"
                )
        return SideImpact(
            after_roster=roster,
            slot_notes=slot_notes,
            delta=delta,
            perceived=perceived,
            upgrades=upgrades,
            drops=drops,
            lost_starters=lost,
            redundant=redundant,
            solves_hole=solves,
            # never give up starting depth for a bench piece at a position where you're already elite
            blocked=bool(redundant) and bool(lost),
        )

    def _upgrades(self, side: Side, incoming_starters, out_ids, post_starters):
        """Which of the side's starters each incoming starter displaces (None = waiver-filled slot)"""
        pool = sorted(
            (p for p in side.team.players
             if id(p) in side.starters and id(p) not in out_ids and id(p) not in post_starters),
            key=lambda p: p.base,
        )
        upgrades = []
        for p in sorted(incoming_starters, key=lambda p: p.base, reverse=True):
            match = next((d for d in pool if d.position == p.position), None) or (pool[0] if pool else None)
            if match is not None:
                pool.remove(match)
                gain = max(0.0, p.base - match.base)
            else:
                gain = max(0.0, p.base - self.drv.get(p.position, 0.0))
            upgrades.append((p, match, gain))
        return upgrades

    def evaluate(self, send: List[PlayerValue], receive: List[PlayerValue]) -> TradeProposal:
        """Score `send` (from you) for `receive` (from the target)"""
        recv_ids = {id(p) for p in receive}
        mine = self._side_impact(self.mine, self.my_weak, send, receive, whose="your")
        theirs = self._side_impact(self.theirs, self.their_weak, receive, send)

        alpha = [p for p in receive if id(p) in self.s_tier]
        tap = TAP_RATE * sum(p.base for p in alpha)
        if len(send) > len(receive):
            tap *= TAP_PACKAGE_MULT      # a depth package for a star is the classic rejection
        bcp = BCP_RATE * sum(value_over_replacement(p, self.drv) for p in theirs.drops) if len(send) >= 2 else 0.0
        # Structure is judged on real pieces. What the target receives only counts if it would start
        # for them: a player who beats a waiver pickup but sits on their bench (Corum behind Warren)
        # is a throw-in, not a player.
        real_in = [p for p, _, _ in theirs.upgrades]
        real_out = [
            p for p in receive
            if value_over_replacement(p, self.drv) >= REAL_MARGIN or id(p) in self.theirs.starters
        ]
        # The effective structure can only make things worse: throw-ins the target receives still
        # clutter its roster, so a 2-for-1 with a bench second piece keeps its nominal penalty.
        tsp = max(structure_penalty(len(send), len(receive)), structure_penalty(len(real_in), len(real_out)))

        # Positional Loss Tax: the target surrenders a starter it can't replace from what comes back
        plt = 0.0
        incoming_starter_positions = {p.position for p, _, _ in theirs.upgrades}
        for p in receive:
            if id(p) not in self.theirs.starters or p.position in incoming_starter_positions:
                continue
            over = max(0.0, p.base - self.drv.get(p.position, 0.0))
            if p.position in self.single and p.base >= max(
                (q.base for q in self.theirs.team.players if id(q) in self.theirs.starters and q.position == p.position),
                default=0.0,
            ):
                plt += PLT_QB_TE_RATE * over      # starting QB1 / TE1 in a 1QB / 1TE format
            elif id(p) in self.s_tier:
                plt += VOID_RATE * over           # a premier starter leaves a real hole

        surrendered_starters = [p for p in receive if id(p) in self.theirs.starters]
        incoming_starters = [p for p, _, _ in theirs.upgrades]
        ap = AP_PENALTY if (
            len(surrendered_starters) >= 2 and len(send) >= 2 and len(incoming_starters) < len(surrendered_starters)
        ) else 0.0

        lat = LATERAL_TAX if sorted(p.position for p in send) == sorted(p.position for p in receive) else 0.0

        # Talent floor: don't hand over a clearly better player for a lesser one, unless a real
        # hole at a different position is being filled
        best_out, best_in = max(p.base for p in receive), max(p.base for p in send)
        tfl = 0.0
        if best_in < TALENT_FLOOR_RATIO * best_out:
            top_out = max(receive, key=lambda p: p.base)
            fills_other_hole = theirs.solves_hole and any(p.position != top_out.position for p in send)
            tfl = (TALENT_GAP_HOLE_RATE if fills_other_hole else TALENT_GAP_RATE) * (best_out - best_in)

        # Sell-low: don't give away your best player for a clearly lesser one just because your own
        # lineup happens to have a surplus at his position (a spare QB, a deep RB room)
        my_best_out, my_best_in = max(p.base for p in send), max(p.base for p in receive)
        sell_low = (
            SELL_LOW_RATE * (my_best_out - my_best_in) if my_best_in < TALENT_FLOOR_RATIO * my_best_out else 0.0
        )

        t = TradeProposal(
            target_id=self.theirs.team.team_id,
            target_name=self.theirs.team.name,
            send=list(send),
            receive=list(receive),
            mine=mine,
            theirs=theirs,
            tap=tap, bcp=bcp, tsp=tsp, plt=plt, ap=ap, lat=lat, tfl=tfl, sell_low=sell_low, real_in=len(real_in), real_out=len(real_out), real_in_players=real_in,
            alpha_assets=alpha,
            notes=_notes(send, receive),
        )
        t.tier = self._assign_tier(t)
        return t

    def attach_lineups(self, t: TradeProposal) -> None:
        """Fill in the slot-by-slot lineups before and after (only for trades that get displayed)"""
        slots = self.snapshot.slots
        for side, impact in ((self.mine, t.mine), (self.theirs, t.theirs)):
            impact.lineup_before = lineup_rows(side.team.players, slots, self.ctx, self.n)
            impact.lineup_after = lineup_rows(impact.after_roster, slots, self.ctx, self.n)

    def _assign_tier(self, t: TradeProposal) -> Optional[str]:
        if t.blocked or t.theirs.delta < 0 or t.my_effective_gain <= 0:
            return None
        mai, mine = t.mai, t.my_effective_gain
        simple = (len(t.send), len(t.receive)) in ((1, 1), (2, 2))
        mutual_holes = t.mine.solves_hole and t.theirs.solves_hole
        no_redundancy = not t.mine.redundant and not t.theirs.redundant
        if mai >= WIN_WIN[1] and mine >= WIN_WIN[3] and simple and mutual_holes and no_redundancy:
            return WIN_WIN[0]
        # a trade that clears the Win-Win numbers but breaks its structure rules is still worth a shot
        if mai >= WORTH_A_SHOT[1] and mine >= self.min_gain:
            return WORTH_A_SHOT[0]
        if LONG_SHOT[1] <= mai < LONG_SHOT[2] and mine >= LONG_SHOT[3]:
            return LONG_SHOT[0]
        return None


def blocked_reason(t: TradeProposal) -> str:
    """Which side would be giving up a starter for a bench piece at a position it's already elite at"""
    parts = []
    for whose, side in (("You", t.mine), ("They", t.theirs)):
        if not side.blocked:
            continue
        gives = " + ".join(p.name for p in side.lost_starters)
        gets = " + ".join(p.name for p in side.redundant)
        pos = "/".join(sorted({p.position for p in side.redundant}))
        verb = "You'd" if whose == "You" else "They'd"
        own = "you already have" if whose == "You" else "they already have"
        outcome = (
            "so this doesn't improve your team." if whose == "You"
            else "so they have no reason to accept."
        )
        parts.append(
            f"Blocked: {verb} give up starter {gives} for bench piece {gets} at {pos}, "
            f"where {own} an elite starter, {outcome}"
        )
    return " ".join(parts)


def why_no_tier(t: TradeProposal) -> str:
    """Which bar a trade misses, in plain words (empty when it does clear a tier)"""
    if t.tier is not None:
        return ""
    if t.blocked:
        return blocked_reason(t)
    if t.theirs.delta < 0:
        return f"Their lineup gets worse ({t.theirs.delta:+.1f} PPG), so they have no reason to accept."
    if t.sell_low and t.my_effective_gain <= 0:
        return (
            f"You'd be selling {max(t.send, key=lambda p: p.base).name} ({max(p.base for p in t.send):.1f} PPG) "
            f"for a much lesser {max(t.receive, key=lambda p: p.base).name} ({max(p.base for p in t.receive):.1f}). "
            f"Your lineup barely changes ({t.mine.delta:+.1f}), but you'd be giving away value (sell-low {-t.sell_low:+.1f})."
        )
    if t.my_effective_gain <= 0:
        return f"It doesn't improve your lineup ({t.mine.delta:+.1f} PPG)."
    mai, mine = t.mai, t.my_effective_gain
    if mai < LONG_SHOT[1]:
        return f"Their acceptance score is {mai:+.1f}, below the {LONG_SHOT[1]:+.1f} floor for even a Long Shot."
    if mai >= WIN_WIN[1]:
        return (
            f"It's easy for them to accept (MAI {mai:+.1f}), but your gain of {mine:+.1f} PPG is below "
            f"the {WORTH_A_SHOT[3]:.1f} needed for Worth a Shot ({WIN_WIN[3]:.1f} for Win-Win)."
        )
    if mai >= WORTH_A_SHOT[1]:
        return (
            f"Their MAI is fine ({mai:+.1f}), but your gain of {mine:+.1f} PPG is below the "
            f"{WORTH_A_SHOT[3]:.1f} needed for Worth a Shot."
        )
    return (
        f"Their MAI is only {mai:+.1f}, which makes it a Long Shot at best, and a Long Shot needs a "
        f"gain of {LONG_SHOT[3]:.1f}+ PPG for you (you'd get {mine:+.1f})."
    )


def _notes(send, receive) -> List[str]:
    notes = []
    for p in list(send) + list(receive):
        if p.injured:
            notes.append(f"{p.name} is {p.injury.replace('_', ' ').title()}")
        elif p.weekly and p.weekly.count(0.0):
            notes.append(f"{p.name} has a bye in the window")
    return notes


# ------------------------------------------------------------------ generation

NEAR_MISS_LIMIT = 3


class TradeResults(dict):
    """{tier: proposals} plus the closest misses and a count of why candidates were rejected"""

    near_misses: List[TradeProposal]
    funnel: Dict[str, int]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.near_misses = []
        self.funnel = {}


FUNNEL_LABELS = {
    "checked": "possible trades checked",
    "blocked": "blocked (a starter for a redundant bench piece)",
    "their_lineup_worse": "make their lineup worse",
    "no_gain_for_you": "don't improve yours",
    "below_floor": "far too hard for them to accept",
    "your_gain_too_small": "fair for both but your gain is under the tier bar",
    "qualified": "cleared a tier",
}


def _funnel_reason(t: TradeProposal) -> str:
    if t.tier is not None:
        return "qualified"
    if t.blocked:
        return "blocked"
    if t.theirs.delta < 0:
        return "their_lineup_worse"
    if t.my_effective_gain <= 0:
        return "no_gain_for_you"
    if t.mai < LONG_SHOT[1]:
        return "below_floor"
    return "your_gain_too_small"


def generate_trades(
    snapshot: LeagueSnapshot,
    my_id: int,
    target_id: int,
    strict: bool = False,
    per_tier: int = PER_TIER,
    min_gain: float = WORTH_A_SHOT[3],
) -> TradeResults:
    """Proposals in each acceptance tier for one target team, best mutual utility first.

    strict=True keeps only trades the target clearly clears (MAI >= +1.5). The result also carries
    `near_misses` (fair trades that fall just short of a tier bar) and a `funnel` of rejection counts
    so an empty result can be explained.
    """
    result = TradeResults({t: [] for t in TIERS})
    if my_id not in snapshot.teams or target_id not in snapshot.teams or my_id == target_id:
        return result

    tc = TradeContext(snapshot, my_id, target_id, min_gain=min_gain)
    if tc.n == 0:
        return result

    def pool(players):
        eligible = (
            p for p in players
            if p.base > 0 and p.position not in NON_TRADE_POSITIONS and p.injury != "INJURY_RESERVE"
        )
        return sorted(eligible, key=lambda p: p.base, reverse=True)[:POOL_SIZE]

    my_pool, their_pool = pool(tc.mine.team.players), pool(tc.theirs.team.players)
    candidates: Dict[str, List[TradeProposal]] = {t: [] for t in TIERS}
    misses: List[TradeProposal] = []
    funnel: Dict[str, int] = {key: 0 for key in FUNNEL_LABELS}
    for send_n, recv_n in [(1, 1), (2, 1), (1, 2), (2, 2)]:
        for send in combinations(my_pool, send_n):
            for receive in combinations(their_pool, recv_n):
                t = tc.evaluate(list(send), list(receive))
                funnel["checked"] += 1
                reason = _funnel_reason(t)
                funnel[reason] += 1
                if reason == "your_gain_too_small":
                    misses.append(t)
                if t.tier is None or (strict and t.mai < STRICT_MIN_MAI):
                    continue
                candidates[t.tier].append(t)
    result.funnel = funnel

    # closest misses: fair for both sides, but the gain for you is under the lowest tier bar
    misses.sort(key=lambda t: (-t.nmu, -t.mai))
    used_send, used_recv = set(), set()
    for t in misses:
        s_ids, r_ids = {id(p) for p in t.send}, {id(p) for p in t.receive}
        if s_ids & used_send or r_ids & used_recv:
            continue
        t.you_why, t.them_why = explain(t)
        tc.attach_lineups(t)
        result.near_misses.append(t)
        used_send |= s_ids
        used_recv |= r_ids
        if len(result.near_misses) >= NEAR_MISS_LIMIT:
            break

    for tier, found in candidates.items():
        found.sort(key=lambda t: (-t.nmu, -t.mai))
        used_send, used_recv = set(), set()
        for t in found:
            s_ids, r_ids = {id(p) for p in t.send}, {id(p) for p in t.receive}
            if s_ids & used_send or r_ids & used_recv:
                continue
            t.you_why, t.them_why = explain(t)
            tc.attach_lineups(t)
            result[tier].append(t)
            used_send |= s_ids
            used_recv |= r_ids
            if len(result[tier]) >= per_tier:
                break
    return result


def evaluate_custom(snapshot: LeagueSnapshot, my_id: int, target_id: int, send, receive) -> TradeProposal:
    """Score a specific trade you typed in (same math as the generated ones)"""
    tc = TradeContext(snapshot, my_id, target_id)
    t = tc.evaluate(list(send), list(receive))
    t.you_why, t.them_why = explain(t)
    tc.attach_lineups(t)
    return t


# ----------------------------------------------------------------- explanations

def _names(players) -> str:
    return " + ".join(p.name for p in players)


def positions_fixed(side: SideImpact) -> List[str]:
    return sorted({p.position for p, _, _ in side.upgrades})


def displaced_players(side: SideImpact) -> List[PlayerValue]:
    return [d for _, d, _ in side.upgrades if d is not None]


def _upgrade_text(side: SideImpact) -> str:
    parts = []
    for p, displaced, gain in side.upgrades:
        over = (
            f"over {displaced.name} ({displaced.base:.1f})"
            if displaced else f"in a slot where {side.slot_notes.get(id(p), 'a waiver pickup would otherwise play')}"
        )
        parts.append(f"{p.name} ({p.base:.1f} PPG) starts at {p.position} {over}, +{gain:.1f} PPG")
    return "; ".join(parts)


PENALTY_LABELS = {
    "TAP": "Alpha tax", "BCP": "bench clutter", "TSP": "trade structure",
    "PLT": "positional loss", "AP": "asymmetry", "LAT": "lateral swap", "TFL": "talent gap",
}


def explain(t: TradeProposal) -> Tuple[str, str]:
    """(why it works for you, why it works for them)"""
    you = []
    if t.mine.upgrades:
        you.append(_upgrade_text(t.mine))
    if t.mine.lost_starters:
        you.append(f"you give up {_names(t.mine.lost_starters)} from your lineup and still net {t.mine.delta:+.1f} PPG")
    if t.mine.drops:
        you.append(f"you'd cut {_names(t.mine.drops)} to make room")
    if t.sell_low:
        you.append(
            f"caution: you'd be selling {max(t.send, key=lambda p: p.base).name} for a much lesser "
            f"{max(t.receive, key=lambda p: p.base).name} (sell-low -{t.sell_low:.1f})"
        )
    dup = [pos for pos in ("QB", "TE") if pos in {p.position for p in t.mine.redundant}]
    you.append(
        "avoids QB/TE duplication" if not t.mine.redundant
        else f"note: {_names(t.mine.redundant)} is redundant behind your top {'/'.join(dup) or 'starter'}"
    )

    them = []
    if t.theirs.upgrades:
        them.append(_upgrade_text(t.theirs))
    else:
        them.append(f"their lineup changes by {t.theirs.delta:+.1f} PPG")
    if t.real_in != len(t.send) or t.real_out != len(t.receive):
        throw_ins = [p.name for p in t.send if p.name not in {q.name for q in t.real_in_players}]
        them.append(
            f"judged on real pieces it's a {t.real_in}-for-{t.real_out} for them"
            + (f" ({', '.join(throw_ins)} wouldn't crack their starting lineup)" if throw_ins else "")
        )
    if t.penalties:
        friction = ", ".join(f"{PENALTY_LABELS[name]} -{value:.1f}" for name, value in t.penalties)
        them.append(f"friction they'll feel: {friction}")
    else:
        them.append("no friction penalties apply")
    if t.alpha_assets:
        pitch = []
        for p in t.alpha_assets:
            if p.injured:
                pitch.append(f"{p.name} is {p.injury.replace('_', ' ').title()}, which is the opening")
            elif p.weekly and p.weekly.count(0.0):
                pitch.append(f"{p.name} is on a bye in the window, so his short-term value to them is lower")
        if pitch:
            them.append("pitch: " + "; ".join(pitch))
    return "; ".join(you) + ".", "; ".join(them) + "."


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


def fetch_drv(league, week: int):
    """Waiver PPG at each position from the league's actual free agents.

    The average of the best DRV_TOP_N healthy free agents, so one hot pickup or odd projection
    can't move every gain and penalty. Returns (drv, live, sources): positions whose lookup fails
    fall back to FALLBACK_DRV, live is False unless every position came from real free agents, and
    sources lists the free agents behind each level so the number can be checked by eye.
    """
    drv = dict(FALLBACK_DRV)
    live = True
    sources: Dict[str, List[Tuple[str, float, float, float]]] = {}
    for pos in FALLBACK_DRV:
        try:
            found = []
            for fa in league.free_agents(week=week, size=25, position=pos):
                status = (getattr(fa, "injuryStatus", "") or "ACTIVE").upper()
                if status in ("OUT", "INJURY_RESERVE", "SUSPENSION"):
                    continue
                avg = getattr(fa, "avg_points", 0) or 0.0
                proj = getattr(fa, "projected_avg_points", 0) or 0.0
                blended = base_points(avg, proj, getattr(fa, "total_points", 0))
                if blended > 0:
                    found.append((getattr(fa, "name", "?"), blended, avg, proj))
            top = sorted(found, key=lambda f: f[1], reverse=True)[:DRV_TOP_N]
            if top:
                drv[pos] = sum(f[1] for f in top) / len(top)
                sources[pos] = top
            else:
                live = False
        except Exception as e:
            logger.warning(f"Could not load free agents for {pos}: {e}")
            live = False
    return drv, live, sources


def build_league_snapshot(league, weeks_ahead: int = WEEKS_AHEAD) -> LeagueSnapshot:
    """Read an espn_api League into plain data the engine can work on"""
    current = getattr(league, "current_week", None) or getattr(league, "nfl_week", 1)
    weeks = list(range(current, min(current + weeks_ahead, SEASON_LAST_WEEK + 1)))
    ratings = _fetch_ratings(league, weeks)
    drv, drv_live, drv_sources = fetch_drv(league, current)

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
        drv_sources=drv_sources,
    )
