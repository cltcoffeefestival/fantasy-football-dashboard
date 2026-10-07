"""
Trade engine: would the OTHER manager realistically accept this, and does it help me?

Every trade is scored for both sides with the same three ideas, each charged exactly once:

  LINEUP   change in the side's starting lineup over the next few weeks (PPG). Rostered players
           always start when available; waivers only fill a slot nobody on the roster can fill
           (bye, injury, no player at the position). This is the concrete, this-month effect.
  VALUE    change in the asset value the side holds. A player's asset value is his season-long
           PPG above the LAST STARTER at his position in this league (not the waiver wire), with
           a convex premium so one star is worth more than two average starters. Incoming players
           a side could rarely start count for less; a human discounts what sits on the bench.
           The target also overvalues what it gives up (endowment), the way real owners do.
  FRICTION the target's cost of a complex deal: more players means more hassle, and a deal that
           frees a roster spot is slightly welcome. Applied to the target only.

  your gain      = LINEUP(you) + VALUE_WEIGHT * VALUE(you)
  acceptance     = LINEUP(them) + VALUE_WEIGHT * VALUE(them) - FRICTION
  trade tiers    require both to clear a bar; Win-Win is reserved for 1-for-1s, the only shape
                 real managers in this league have entertained so far.

Base (season) value and short-term lineup value are kept separate: matchups, byes and injury
status move the 4-week lineup numbers; the asset value uses the season blend, discounted only
for injuries long enough to matter beyond the window.
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
# How the league's flex starts are typically split, for counting how many players start at a position
FLEX_SHARE = {"RB": 0.5, "WR": 0.4, "TE": 0.1}
SUPERFLEX_SHARE = {"QB": 0.9, "RB": 0.05, "WR": 0.05}

# Probability a player is available in each upcoming week, by ESPN injury status.
# Indexed by weeks from now; beyond the list the player is treated as healthy.
INJURY_AVAILABILITY = {
    "QUESTIONABLE": [0.9],
    "DOUBTFUL": [0.4, 0.8],
    "OUT": [0.0, 0.3, 0.6, 0.85],
    "INJURY_RESERVE": [0.0, 0.0, 0.2, 0.4, 0.6, 0.8],
    "SUSPENSION": [0.0, 0.4, 0.8],
}
ASSET_HORIZON = 8            # weeks over which an injury discounts a player's season-long value
AVAILABLE_FRACTION = 0.5     # a rostered player starts in a week he is expected to play at least this much
WAIVER_BEATS_BY = 3.0        # ...unless a waiver pickup would clearly outscore him by this much (then anyone streams)

# Matchup: ESPN gives each opponent a rank vs. a position (1..32). We assume 1 is the toughest
# defense and 32 the easiest. Only applied when there is no ESPN weekly projection (which already
# reflects the opponent), so a matchup never counts twice.
MATCHUP_SWING = 0.08
MATCHUP_POSITIONS = {"QB": "1", "RB": "2", "WR": "3", "TE": "4"}  # ESPN defaultPositionId

# ---- value
CONVEXITY = 10.0             # asset = v * (1 + v / CONVEXITY): stars are worth more than the sum of parts
# Managers don't value a player purely over the last starter; part of his raw scoring counts
# regardless of position (which is why a 16-PPG QB fetches more than value-over-replacement says).
RAW_POINTS_ANCHOR = 0.25
VALUE_WEIGHT = 0.5           # PPG-equivalent of one point of asset value
ENDOWMENT = 1.15             # the target overvalues what it gives up by this much
# A bench player still has trade value, but a human discounts what he can't start. Single-slot
# positions (QB/TE in 1QB/1TE formats) are discounted hardest: a second QB is nearly dead weight.
BENCH_WEIGHT = {"QB": 0.25, "TE": 0.3}
BENCH_WEIGHT_DEFAULT = 0.5
# ---- friction (target only). Calibrated on real outcomes: every multi-player offer tried so far
# was rejected (7 of 7) while the simple 1-for-1s were entertained.
HASSLE = {2: 0.0, 3: 1.5, 4: 2.5}
# A same-position swap has to be clearly better to be worth the target's bother; a sideways move
# ("my RB for your slightly better RB") is the trade nobody makes.
LATERAL_MIN_GAIN = 2.5
# ---- balance. A trade can add up to a gain while leaving a position that was already below the
# league average even thinner: no depth there, one injury from a hole. Each PPG a below-average
# position sinks further (relative to the league average) is charged this much on top of the lineup
# change. Applies to both sides; only deepening an existing weakness counts, not trimming a strength.
WEAKNESS_RATE = 0.5
FREED_SPOT_BONUS = 0.5       # per roster spot the target frees (consolidation is welcome)
NON_TRADE_POSITIONS = {"K", "D/ST"}   # nobody trades for these; keep them out of the candidate pools
S_TIER_TOP = {"QB": 3, "RB": 5, "WR": 6, "TE": 3}   # the league's star tier, for explanations
WEAK_SLOT_MARGIN = 2.0       # a starter within this many PPG of the last-starter level is a weak slot
STRENGTH_MARGIN = 0.10       # +/-10% vs. league average starters => strength / weakness
DRV_TOP_N = 3                # waiver level = average of the best few free agents, not the single best

# ---- tiers: (name, min acceptance, min your gain)
WIN_WIN = ("Win-Win", 1.5, 1.5)
DEFAULT_MIN_GAIN = 0.3       # your smallest gain for a Worth a Shot trade unless the slider says otherwise
WORTH_A_SHOT = ("Worth a Shot", 0.5, 1.0)
LONG_SHOT = ("Long Shot", -3.0, 2.0)
STRICT_MIN_ACCEPT = 1.5      # the "strict acceptance" filter: only trades the target clearly clears
TIERS = ["Win-Win", "Worth a Shot", "Long Shot"]
PER_TIER = 3
POOL_SIZE = 8                # players per side considered in a swap

# Used only when the free agent lookup fails: rough full-PPR waiver-wire PPG by position
FALLBACK_DRV = {"QB": 13.0, "RB": 8.5, "WR": 9.5, "TE": 6.5, "D/ST": 6.0, "K": 7.0}


@dataclass
class PlayerValue:
    name: str
    position: str
    pro_team: str
    injury: str
    base: float                # blended season points per game (talent)
    weekly: List[float] = field(default_factory=list)   # expected points in each week of the window

    @property
    def avg(self) -> float:
        return sum(self.weekly) / len(self.weekly) if self.weekly else 0.0

    @property
    def injured(self) -> bool:
        return self.injury not in ("", "ACTIVE", "NORMAL", "NONE")

    @property
    def long_term_availability(self) -> float:
        """Share of the next ASSET_HORIZON weeks the player is expected to be available"""
        return sum(availability(self.injury, i) for i in range(ASSET_HORIZON)) / ASSET_HORIZON


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

    @property
    def levels(self) -> Dict[str, float]:
        """Last-starter PPG at each position (the replacement level asset value is measured against)"""
        return starter_levels(self.teams, self.slots, self.drv)


@dataclass
class SideImpact:
    """What a trade does to one team, in the three currencies the engine uses"""
    lineup_delta: float = 0.0                            # starting PPG change over the window
    value_in: float = 0.0                                # asset value received (bench-discounted)
    value_out: float = 0.0                               # asset value given up (plus any cuts)
    value_delta: float = 0.0                             # in - out (times ENDOWMENT for the target)
    hassle: float = 0.0                                  # package friction (target only)
    lateral: float = 0.0                                 # sideways-swap friction (target only)
    freed: float = 0.0                                   # roster-spot bonus (target only)
    weakness: float = 0.0                                # charge for deepening a below-average position
    # (position, PPG before, PPG after, league average) for each position the trade weakens further
    weakened: List[Tuple[str, float, float, float]] = field(default_factory=list)
    shape: str = ""                                      # consolidation / diversification / depth / upgrade / ...
    # (incoming starter, starter he displaces or None for an empty slot, PPG gained at the slot)
    upgrades: List[Tuple[PlayerValue, Optional[PlayerValue], float]] = field(default_factory=list)
    start_share: Dict[int, float] = field(default_factory=dict)   # id(player) -> share of weeks he starts (after)
    drops: List[PlayerValue] = field(default_factory=list)       # bench players cut to make room
    outgoing: List[PlayerValue] = field(default_factory=list)    # what the side gives up
    lost_starters: List[PlayerValue] = field(default_factory=list)
    redundant: List[PlayerValue] = field(default_factory=list)   # incoming players who would rarely start
    best_out: Optional[PlayerValue] = None               # the most valuable player the side gives up
    best_out_share: float = 0.0                          # how often that player started for the side before
    best_in: Optional[PlayerValue] = None
    after_roster: List[PlayerValue] = field(default_factory=list)   # lineup-eligible roster after the trade
    lineup_before: List[Tuple[str, str, float]] = field(default_factory=list)
    lineup_after: List[Tuple[str, str, float]] = field(default_factory=list)
    slot_notes: Dict[int, str] = field(default_factory=dict)     # id(incoming) -> why a slot counts as empty
    need_solved: List[str] = field(default_factory=list)         # positions where a weak/empty slot was fixed
    risks: List[str] = field(default_factory=list)               # injuries, byes, long-term doubts in what comes in
    blocked: bool = False                                        # gives up a starter for nothing it could start

    @property
    def delta(self) -> float:
        return self.lineup_delta

    @property
    def friction(self) -> float:
        return self.hassle + self.lateral

    @property
    def total(self) -> float:
        return self.lineup_delta + VALUE_WEIGHT * self.value_delta - self.friction + self.freed - self.weakness


@dataclass
class TradeProposal:
    target_id: int
    target_name: str
    send: List[PlayerValue]          # what you send to the target
    receive: List[PlayerValue]       # what the target sends you
    mine: SideImpact
    theirs: SideImpact
    min_gain: float = DEFAULT_MIN_GAIN   # the smallest gain for you that this evaluation treated as Worth a Shot
    alpha_assets: List[PlayerValue] = field(default_factory=list)   # league stars the target gives up
    tier: Optional[str] = None
    confidence: str = ""
    works_because: str = ""
    fails_because: str = ""
    notes: List[str] = field(default_factory=list)
    you_why: str = ""
    them_why: str = ""

    @property
    def acceptance(self) -> float:
        """How a normal manager on the other side would feel about this (PPG-equivalent)"""
        return self.theirs.total

    mai = acceptance             # old name, still used by the UI

    @property
    def my_gain(self) -> float:
        """Your starting lineup change (PPG)"""
        return self.mine.lineup_delta

    @property
    def my_effective_gain(self) -> float:
        """Your lineup change plus the asset value you net: what the trade is really worth to you"""
        return self.mine.total

    @property
    def delta_lineup(self) -> float:
        """The target's net lineup change"""
        return self.theirs.lineup_delta

    @property
    def nmu(self) -> float:
        """Net Mutual Utility: your gain plus their acceptance"""
        return self.my_effective_gain + self.acceptance

    @property
    def blocked(self) -> bool:
        return self.mine.blocked or self.theirs.blocked

    @property
    def penalties(self) -> List[Tuple[str, float]]:
        """The target's frictions that actually apply, as (label, PPG)"""
        named = []
        if self.theirs.value_delta < 0:
            named.append(("value lost", -VALUE_WEIGHT * self.theirs.value_delta))
        if self.theirs.hassle:
            named.append(("package hassle", self.theirs.hassle))
        if self.theirs.lateral:
            named.append(("sideways swap", self.theirs.lateral))
        return named

    @property
    def simple(self) -> bool:
        return len(self.send) == 1 and len(self.receive) == 1


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
        projected = (stats.get(week) or {}).get("projected_points")
        value = projected or base
        if game and not projected:        # ESPN's weekly projection already prices the opponent
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
    """Each slot with the waiver pickup that would fill it if nobody on the roster can"""
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


def _available(p: PlayerValue, week: int) -> bool:
    """A rostered player starts in a week he is expected to play; byes and injuries sit him"""
    return p.weekly[week] > 0 and p.weekly[week] >= AVAILABLE_FRACTION * p.base


def _solve_week(players: List[PlayerValue], ctx, week: int):
    """Best lineup for one week. Returns (points, players who start, per-slot assignments).

    A slot goes to the best available rostered player. It is filled by a waiver pickup only when
    nobody eligible is available (bye, injury, no player at the position) or the best rostered option
    is so far below the waiver level that any manager would stream instead.
    """
    ranked = sorted(players, key=lambda p: p.weekly[week], reverse=True)
    used = set()
    total = 0.0
    starters = []
    assigned = []   # per slot: (player or None for a waiver pickup, points)
    for eligible, waiver in ctx:
        best = next(
            (idx for idx, p in enumerate(ranked)
             if idx not in used and p.position in eligible and _available(p, week)),
            None,
        )
        if best is not None and waiver - ranked[best].weekly[week] <= WAIVER_BEATS_BY:
            p = ranked[best]
            used.add(best)
            total += p.weekly[week]
            starters.append(p)
            assigned.append((p, p.weekly[week]))
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


def position_ppg(players: List[PlayerValue], dedicated: Dict[str, int], drv: Dict[str, float], n_weeks: int) -> Dict[str, float]:
    """Strength at each position: weekly points of the best players available for its dedicated
    slots (2 RB, 2 WR, 1 TE...), averaged over the window. An empty slot counts at the waiver level.
    Flex is left out so a position doesn't look stronger or weaker just because of who gets the flex."""
    out = {}
    for pos, k in dedicated.items():
        total = 0.0
        for w in range(n_weeks):
            pts = sorted((p.weekly[w] for p in players if p.position == pos and _available(p, w)), reverse=True)[:k]
            pts += [drv.get(pos, 0.0)] * (k - len(pts))
            total += sum(max(x, 0.0) for x in pts)
        out[pos] = total / n_weeks
    return out


def start_counts(players: List[PlayerValue], ctx, n_weeks: int) -> Dict[int, int]:
    """How many weeks of the window each player is in the optimal lineup"""
    counts: Dict[int, int] = {}
    for w in range(n_weeks):
        for p in _solve_week(players, ctx, w)[1]:
            counts[id(p)] = counts.get(id(p), 0) + 1
    return counts


def start_shares(players: List[PlayerValue], ctx, n_weeks: int) -> Dict[int, float]:
    """Share of the window each player starts (0..1); a continuous starter/bench measure"""
    return {pid: c / n_weeks for pid, c in start_counts(players, ctx, n_weeks).items()}


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


# ---------------------------------------------------------------- asset value

def starter_levels(teams: Dict[int, TeamSnapshot], slots, drv: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """PPG of the last starter at each position across the league.

    K = teams x (dedicated slots + the position's usual share of flex slots); the level is the K-th
    best rostered player's season PPG. This is the replacement level a manager feels: "is he a
    starter in this league?" It scales with league size and format (superflex raises QB demand).
    It is never below the waiver level: if a free agent outscores the league's last starter, the
    free agent is the last starter.
    """
    drv = drv or {}
    n_teams = max(1, len(teams))
    demand: Dict[str, float] = {}
    for _, eligible in slots:
        if len(eligible) == 1:
            pos = next(iter(eligible))
            demand[pos] = demand.get(pos, 0.0) + 1
        else:
            share = SUPERFLEX_SHARE if "QB" in eligible else FLEX_SHARE
            for pos, s in share.items():
                if pos in eligible:
                    demand[pos] = demand.get(pos, 0.0) + s
    levels = {}
    for pos, d in demand.items():
        k = max(1, round(n_teams * d))
        ranked = sorted((p.base for t in teams.values() for p in t.players if p.position == pos), reverse=True)
        rostered = ranked[k - 1] if len(ranked) >= k else (ranked[-1] if ranked else 0.0)
        levels[pos] = max(rostered, drv.get(pos, 0.0))
    return levels


def value_over_replacement(player: PlayerValue, drv: Dict[str, float]) -> float:
    """PPG above a waiver pickup at the position (shown in the UI; not used to score trades)"""
    return max(0.0, player.base - drv.get(player.position, 0.0))


def asset_value(player: PlayerValue, levels: Dict[str, float]) -> float:
    """Season-long value of a player as an asset: PPG above the league's last starter (managers also
    anchor a little on raw points), convex so stars are worth more than the sum of parts, and
    discounted for an injury that outlasts the window"""
    v = max(0.0, player.base - (1 - RAW_POINTS_ANCHOR) * levels.get(player.position, 0.0))
    return v * (1 + v / CONVEXITY) * player.long_term_availability


def s_tier_ids(snapshot: LeagueSnapshot) -> set:
    """Ids of the league's star tier: the top few rostered players at each position"""
    ids = set()
    for pos, top in S_TIER_TOP.items():
        ranked = sorted(
            (p for t in snapshot.teams.values() for p in t.players if p.position == pos),
            key=lambda p: p.base,
            reverse=True,
        )
        ids.update(id(p) for p in ranked[:top])
    return ids


def hassle(n_players: int) -> float:
    """The target's friction for a deal of this many total players"""
    return HASSLE.get(n_players, HASSLE[max(HASSLE)])


# -------------------------------------------------------------------- scoring

class Side:
    """One team's pre-trade facts"""

    def __init__(self, team: TeamSnapshot, tc: "TradeContext"):
        self.team = team
        self.before = lineup_ppg(team.players, tc.ctx, tc.n)
        self.pos_before = position_ppg(team.players, tc.dedicated, tc.drv, tc.n)
        self.shares = start_shares(team.players, tc.ctx, tc.n)
        self.starters = {pid for pid, s in self.shares.items() if s >= 0.5}
        # cheapest to cut first: bench before starters, then lowest asset value
        self.cut = sorted(
            team.players,
            key=lambda p: (self.shares.get(id(p), 0.0), asset_value(p, tc.levels), p.base),
        )


class TradeContext:
    """Everything about one (you, target) pair that doesn't change between candidate trades"""

    def __init__(self, snapshot: LeagueSnapshot, my_id: int, target_id: int, min_gain: float = DEFAULT_MIN_GAIN):
        self.snapshot = snapshot
        self.min_gain = min_gain     # your smallest gain for a Worth a Shot trade (PPG)
        self.n = len(snapshot.weeks)
        self.drv = snapshot.drv
        self.levels = snapshot.levels
        self.ctx = slot_context(snapshot.slots, snapshot.drv)
        self.s_tier = s_tier_ids(snapshot)
        self.single = single_slot_positions(snapshot.slots)
        # dedicated starting slots per position (flex left out), and the league average strength there
        self.dedicated: Dict[str, int] = {}
        for _, eligible in snapshot.slots:
            if len(eligible) == 1:
                pos = next(iter(eligible))
                if pos not in NON_TRADE_POSITIONS:
                    self.dedicated[pos] = self.dedicated.get(pos, 0) + 1
        by_team = [position_ppg(t.players, self.dedicated, self.drv, self.n) for t in snapshot.teams.values()] if self.n else []
        self.league_pos = {
            pos: sum(b[pos] for b in by_team) / len(by_team) for pos in self.dedicated
        } if by_team else {}
        self.mine = Side(snapshot.teams[my_id], self)
        self.theirs = Side(snapshot.teams[target_id], self)

    def bench_weight(self, pos: str) -> float:
        return BENCH_WEIGHT.get(pos, BENCH_WEIGHT_DEFAULT) if pos in self.single else BENCH_WEIGHT_DEFAULT

    def usefulness(self, p: PlayerValue, share: float) -> float:
        """How much of a player's asset value a side feels, given how often it would start him"""
        w = self.bench_weight(p.position)
        return w + (1 - w) * min(1.0, share)

    # ---- one side of a trade
    def _side_impact(self, side: Side, outgoing, incoming, is_target: bool, whose: str) -> SideImpact:
        out_ids = {id(p) for p in outgoing}
        drops = [p for p in side.cut if id(p) not in out_ids][: max(0, len(incoming) - len(outgoing))]
        drop_ids = {id(p) for p in drops}
        roster = [p for p in side.team.players if id(p) not in out_ids and id(p) not in drop_ids] + list(incoming)

        after = lineup_ppg(roster, self.ctx, self.n)
        shares = start_shares(roster, self.ctx, self.n)
        lineup_delta = after - side.before

        # balance: a position already below the league average that the trade pushes further below
        pos_after = position_ppg(roster, self.dedicated, self.drv, self.n)
        weakened = []
        for pos, avg in self.league_pos.items():
            before_pos, after_pos = side.pos_before.get(pos, 0.0), pos_after.get(pos, 0.0)
            deeper = max(0.0, avg - after_pos) - max(0.0, avg - before_pos)
            if deeper > 0.05:
                weakened.append((pos, before_pos, after_pos, avg))
        weakness = WEAKNESS_RATE * sum(max(0.0, avg - a) - max(0.0, avg - b) for _, b, a, avg in weakened)

        # asset value: what comes in is discounted by how often it would start. For the target, what
        # goes out is discounted the same way (a backup QB he never starts is not his star) and then
        # overvalued a little (endowment). For you, what goes out counts in full: a spare elite player
        # is a trade chip, and the engine's job is to stop you selling him low.
        value_in = sum(asset_value(p, self.levels) * self.usefulness(p, shares.get(id(p), 0.0)) for p in incoming)
        value_out = sum(
            asset_value(p, self.levels) * (self.usefulness(p, side.shares.get(id(p), 0.0)) if is_target else 1.0)
            for p in outgoing
        ) + sum(asset_value(p, self.levels) for p in drops)
        value_delta = value_in - (ENDOWMENT if is_target else 1.0) * value_out

        incoming_starters = [p for p in incoming if shares.get(id(p), 0.0) >= 0.5]
        lost = [p for p in outgoing if id(p) in side.starters]
        redundant = [p for p in incoming if shares.get(id(p), 0.0) < 0.25]
        upgrades = self._upgrades(side, incoming_starters, out_ids, shares)

        slot_notes, need_solved = {}, []
        for p, displaced, _ in upgrades:
            level = self.levels.get(p.position, 0.0)
            if displaced is not None and id(displaced) in out_ids:
                continue      # he takes the slot a traded player leaves: not a hole fixed
            if displaced is None:
                slot_notes[id(p)] = f"{whose} roster has nobody to play, so it was filled from waivers ({self.drv.get(p.position, 0.0):.1f} PPG)"
                need_solved.append(p.position)
            elif displaced.base - level <= WEAK_SLOT_MARGIN:
                need_solved.append(p.position)

        risks = []
        for p in incoming:
            if p.injured:
                risks.append(f"{p.name} is {p.injury.replace('_', ' ').title()}")
            elif p.weekly and p.weekly.count(0.0):
                risks.append(f"{p.name} has a bye in the window")

        n_total = len(incoming) + len(outgoing)
        sideways = sorted(p.position for p in incoming) == sorted(p.position for p in outgoing)
        lateral = 0.0
        if is_target and sideways and lineup_delta < LATERAL_MIN_GAIN:
            lateral = LATERAL_MIN_GAIN - max(0.0, lineup_delta)      # sideways: not worth their bother
        return SideImpact(
            lineup_delta=lineup_delta,
            value_in=value_in,
            value_out=value_out,
            value_delta=value_delta,
            hassle=hassle(n_total) if is_target else 0.0,
            lateral=lateral,
            freed=FREED_SPOT_BONUS * max(0, len(outgoing) - len(incoming)) if is_target else 0.0,
            weakness=weakness,
            weakened=weakened,
            shape=trade_shape(len(outgoing), len(incoming), incoming_starters, redundant, lineup_delta),
            upgrades=upgrades,
            start_share=shares,
            drops=drops,
            outgoing=list(outgoing),
            lost_starters=lost,
            redundant=redundant,
            best_out=max(outgoing, key=lambda p: asset_value(p, self.levels), default=None),
            best_out_share=max((side.shares.get(id(p), 0.0) for p in outgoing), default=0.0),
            best_in=max(incoming, key=lambda p: asset_value(p, self.levels), default=None),
            after_roster=roster,
            slot_notes=slot_notes,
            need_solved=sorted(set(need_solved)),
            risks=risks,
            # a pure dump: give up a starter and get nothing you would ever start
            blocked=bool(lost) and len(redundant) == len(incoming),
        )

    def _upgrades(self, side: Side, incoming_starters, out_ids, shares):
        """Which of the side's starters each incoming starter displaces.

        Starters the side keeps but who no longer start are matched first (a real upgrade over them);
        then starters leaving in the trade (the newcomer takes the slot they vacate); None only when
        the slot really had nobody (it was filled from waivers before).
        """
        benched = sorted(
            (p for p in side.team.players
             if id(p) in side.starters and id(p) not in out_ids and shares.get(id(p), 0.0) < 0.5),
            key=lambda p: p.base,
        )
        leaving = sorted((p for p in side.team.players if id(p) in side.starters and id(p) in out_ids), key=lambda p: p.base)
        ordered = sorted(incoming_starters, key=lambda p: p.base, reverse=True)
        matched: Dict[int, Optional[PlayerValue]] = {}
        # same-position matches first (a TE replaces the benched TE), then anyone left takes what remains
        for pool in (benched, leaving):
            for same_position in (True, False):
                for p in ordered:
                    if id(p) in matched:
                        continue
                    d = next((d for d in pool if not same_position or d.position == p.position), None)
                    if d is not None:
                        pool.remove(d)
                        matched[id(p)] = d
        upgrades = []
        for p in ordered:
            d = matched.get(id(p))
            if d is None:
                upgrades.append((p, None, max(0.0, p.base - self.drv.get(p.position, 0.0))))
            elif id(d) in out_ids:
                upgrades.append((p, d, p.base - d.base))
            else:
                upgrades.append((p, d, max(0.0, p.base - d.base)))
        return upgrades

    def evaluate(self, send: List[PlayerValue], receive: List[PlayerValue]) -> TradeProposal:
        """Score `send` (from you) for `receive` (from the target)"""
        mine = self._side_impact(self.mine, send, receive, is_target=False, whose="your")
        theirs = self._side_impact(self.theirs, receive, send, is_target=True, whose="their")
        t = TradeProposal(
            target_id=self.theirs.team.team_id,
            target_name=self.theirs.team.name,
            send=list(send),
            receive=list(receive),
            mine=mine,
            theirs=theirs,
            min_gain=self.min_gain,
            alpha_assets=[p for p in receive if id(p) in self.s_tier],
            notes=_notes(send, receive),
        )
        t.tier = self._assign_tier(t)
        t.confidence = confidence(t, self.snapshot)
        t.works_because, t.fails_because = headline_reasons(t)
        return t

    def attach_lineups(self, t: TradeProposal) -> None:
        """Fill in the slot-by-slot lineups before and after (only for trades that get displayed)"""
        slots = self.snapshot.slots
        for side, impact in ((self.mine, t.mine), (self.theirs, t.theirs)):
            impact.lineup_before = lineup_rows(side.team.players, slots, self.ctx, self.n)
            impact.lineup_after = lineup_rows(impact.after_roster, slots, self.ctx, self.n)

    def _assign_tier(self, t: TradeProposal) -> Optional[str]:
        if t.blocked or t.my_effective_gain <= 0:
            return None
        accept, mine = t.acceptance, t.my_effective_gain
        if accept >= WIN_WIN[1] and mine >= WIN_WIN[2] and t.simple:
            return WIN_WIN[0]
        if accept >= WORTH_A_SHOT[1] and mine >= self.min_gain:
            return WORTH_A_SHOT[0]
        if LONG_SHOT[1] <= accept < WORTH_A_SHOT[1] and mine >= LONG_SHOT[2]:
            return LONG_SHOT[0]
        return None


def trade_shape(gives: int, gets: int, incoming_starters, redundant, lineup_delta: float) -> str:
    """The roster archetype of a side's end of the deal"""
    starts = len(incoming_starters)
    if gets < gives:
        return "consolidation" if starts else "roster clearing"
    if gets > gives:
        if starts >= 2:
            return "diversification"
        return "depth" if starts == 1 else "bench clutter"
    if gets == len(redundant):
        return "redundant depth"
    return "starter upgrade" if lineup_delta > 0 else "swap"


def confidence(t: TradeProposal, snapshot: LeagueSnapshot) -> str:
    """How much to trust the numbers behind this trade"""
    doubts = 0
    if not snapshot.drv_live:
        doubts += 1
    if not snapshot.schedule_adjusted:
        doubts += 1
    if any(p.injured for p in t.send + t.receive):
        doubts += 1
    if abs(t.acceptance - WORTH_A_SHOT[1]) < 0.75:
        doubts += 1      # right on the line: a small projection change flips it
    return "high" if doubts == 0 else "medium" if doubts == 1 else "low"


def headline_reasons(t: TradeProposal) -> Tuple[str, str]:
    """(the main reason it works, the main reason it might not), from the target's point of view"""
    th = t.theirs
    works = []
    if th.lineup_delta > 0:
        works.append(f"their lineup improves {th.lineup_delta:+.1f} PPG")
    if th.value_delta > 0:
        works.append(f"they come out ahead on value ({VALUE_WEIGHT * th.value_delta:+.1f})")
    if th.need_solved:
        works.append(f"fixes their {'/'.join(th.need_solved)}")
    if th.shape == "consolidation":
        works.append("they consolidate two pieces into one better player")
    fails = []
    if th.blocked:
        fails.append("they'd get nothing they would start")
    if th.lineup_delta < 0:
        fails.append(f"their lineup drops {th.lineup_delta:+.1f} PPG")
    if th.value_delta < 0 and th.best_out is not None and th.best_out_share < 0.5:
        fails.append(f"they give up {th.best_out.name}, even if he was riding their bench")
    elif th.value_delta < 0 and th.best_out is not None:
        fails.append(f"they give up the best player in the deal ({th.best_out.name})"
                     if th.best_in is None or asset_value_cmp(th) else f"they lose value ({VALUE_WEIGHT * th.value_delta:+.1f})")
    if th.hassle:
        fails.append(f"it's a {len(t.send) + len(t.receive)}-player deal, which owners here have rejected every time")
    if th.lateral:
        fails.append("it's a sideways same-position swap that barely moves their lineup")
    if th.redundant:
        fails.append(f"{' + '.join(p.name for p in th.redundant)} would sit on their bench")
    if th.weakened:
        fails.append(f"it thins their already-weak {weakened_text(th)}")
    return ("; ".join(works) or "it doesn't help them", "; ".join(fails) or "nothing obvious")


def weakened_text(side: SideImpact) -> str:
    return ", ".join(f"{pos} ({b:.1f} → {a:.1f} vs league {avg:.1f})" for pos, b, a, avg in side.weakened)


def asset_value_cmp(side: SideImpact) -> bool:
    return side.value_out > side.value_in


def blocked_reason(t: TradeProposal) -> str:
    parts = []
    for whose, side in (("You", t.mine), ("They", t.theirs)):
        if not side.blocked:
            continue
        gives = " + ".join(p.name for p in side.lost_starters)
        gets = " + ".join(p.name for p in side.redundant)
        verb = "You'd" if whose == "You" else "They'd"
        outcome = "so this doesn't improve your team." if whose == "You" else "so they have no reason to accept."
        parts.append(f"Blocked: {verb} give up starter {gives} for {gets}, who would sit on the bench, {outcome}")
    return " ".join(parts)


def why_no_tier(t: TradeProposal) -> str:
    """Which bar a trade misses, in plain words (empty when it does clear a tier)"""
    if t.tier is not None:
        return ""
    if t.blocked:
        return blocked_reason(t)
    if t.my_effective_gain <= 0:
        if t.mine.value_delta < 0 and t.mine.best_out is not None:
            return (
                f"Your lineup changes {t.mine.lineup_delta:+.1f}, and you'd be giving away more value than you get "
                f"({t.mine.best_out.name} is the best player in the deal, value {VALUE_WEIGHT * t.mine.value_delta:+.1f}); "
                f"net {t.my_effective_gain:+.1f} for you."
            )
        if t.mine.weakened and t.mine.lineup_delta > 0:
            return (
                f"Your lineup improves ({t.mine.lineup_delta:+.1f}), but it thins your already-weak "
                f"{weakened_text(t.mine)} (−{t.mine.weakness:.1f}); net {t.my_effective_gain:+.1f} for you."
            )
        return f"It doesn't improve your team ({t.my_effective_gain:+.1f} for you)."
    accept, mine = t.acceptance, t.my_effective_gain
    if accept < LONG_SHOT[1]:
        return f"Their acceptance score is {accept:+.1f}, below the {LONG_SHOT[1]:+.1f} floor for even a Long Shot: {t.fails_because}."
    if accept >= WORTH_A_SHOT[1]:
        return (
            f"It's fine for them ({accept:+.1f}), but your gain of {mine:+.1f} is below the "
            f"{t.min_gain:.1f} needed for Worth a Shot."
        )
    return (
        f"It's a stretch for them ({accept:+.1f}: {t.fails_because}), so it's a Long Shot at best, "
        f"and a Long Shot needs {LONG_SHOT[2]:.1f}+ for you (you'd get {mine:+.1f})."
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

NEAR_MISS_LIMIT = 5
NEAR_MISS_MIN_GAIN = 0.2      # smaller than this isn't an edge worth a trade


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
    "blocked": "blocked (a starter for nothing they'd start)",
    "no_gain_for_you": "don't improve your team",
    "below_floor": "far too hard for them to accept",
    "stretch_for_them": "a stretch for them and not big enough for you to be worth it",
    "your_gain_too_small": "acceptable to them but your gain is under the tier bar",
    "qualified": "cleared a tier",
}


def _funnel_reason(t: TradeProposal) -> str:
    if t.tier is not None:
        return "qualified"
    if t.blocked:
        return "blocked"
    if t.my_effective_gain <= 0:
        return "no_gain_for_you"
    if t.acceptance < LONG_SHOT[1]:
        return "below_floor"
    if t.acceptance < WORTH_A_SHOT[1]:
        return "stretch_for_them"
    return "your_gain_too_small"


def wanted_players(tc: TradeContext, receiver: Side, givers: List[PlayerValue]) -> List[PlayerValue]:
    """The giver's players a side would actually use: those who would start for it, ranked by the
    lineup gain they bring, plus its best assets (consolidation targets). Generation starts from
    what each manager needs, not from the best names available."""
    gains = []
    for p in givers:
        if p.base <= 0 or p.position in NON_TRADE_POSITIONS or p.injury == "INJURY_RESERVE":
            continue
        if p.base <= tc.drv.get(p.position, 0.0):
            continue      # nobody trades for a player they could pick up for free
        roster = list(receiver.team.players) + [p]
        share = start_shares(roster, tc.ctx, tc.n).get(id(p), 0.0)
        gain = lineup_ppg(roster, tc.ctx, tc.n) - receiver.before
        gains.append((p, share, gain))
    useful = sorted((g for g in gains if g[1] >= 0.5), key=lambda g: g[2], reverse=True)
    pool = [p for p, _, _ in useful[:POOL_SIZE]]
    # consolidation: a star the receiver could never "need" by lineup math still moves deals
    stars = sorted((g[0] for g in gains if asset_value(g[0], tc.levels) > 0), key=lambda p: asset_value(p, tc.levels), reverse=True)
    for p in stars[:2]:
        if p not in pool:
            pool.append(p)
    return pool


def generate_trades(
    snapshot: LeagueSnapshot,
    my_id: int,
    target_id: int,
    strict: bool = False,
    per_tier: int = PER_TIER,
    min_gain: float = DEFAULT_MIN_GAIN,
) -> TradeResults:
    """Proposals in each acceptance tier for one target team, best mutual utility first.

    strict=True keeps only trades the target clearly clears (acceptance >= +1.5). The result also
    carries `near_misses` (plausible trades that fall just short of a tier bar) and a `funnel` of
    rejection counts so an empty result can be explained.
    """
    result = TradeResults({t: [] for t in TIERS})
    if my_id not in snapshot.teams or target_id not in snapshot.teams or my_id == target_id:
        return result

    tc = TradeContext(snapshot, my_id, target_id, min_gain=min_gain)
    if tc.n == 0:
        return result

    # what they would use from me, and what I would use from them
    my_pool = wanted_players(tc, tc.theirs, tc.mine.team.players)
    their_pool = wanted_players(tc, tc.mine, tc.theirs.team.players)
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
                if reason in ("your_gain_too_small", "stretch_for_them") and t.my_effective_gain >= NEAR_MISS_MIN_GAIN:
                    misses.append(t)
                if t.tier is None or (strict and t.acceptance < STRICT_MIN_ACCEPT):
                    continue
                candidates[t.tier].append(t)
    result.funnel = funnel

    def take(found, limit, into):
        found.sort(key=lambda t: (-t.nmu, -t.acceptance))
        used_send, used_recv = set(), set()
        for t in found:
            s_ids, r_ids = {id(p) for p in t.send}, {id(p) for p in t.receive}
            if s_ids & used_send or r_ids & used_recv:
                continue
            t.you_why, t.them_why = explain(t)
            tc.attach_lineups(t)
            into.append(t)
            used_send |= s_ids
            used_recv |= r_ids
            if len(into) >= limit:
                break

    take(misses, NEAR_MISS_LIMIT, result.near_misses)
    for tier, found in candidates.items():
        take(found, per_tier, result[tier])
    return result


def evaluate_custom(
    snapshot: LeagueSnapshot, my_id: int, target_id: int, send, receive, min_gain: float = DEFAULT_MIN_GAIN
) -> TradeProposal:
    """Score a specific trade you typed in (same math as the generated ones)"""
    tc = TradeContext(snapshot, my_id, target_id, min_gain=min_gain)
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
    out_ids = {id(p) for p in side.outgoing}
    return [d for _, d, _ in side.upgrades if d is not None and id(d) not in out_ids]


def _upgrade_text(side: SideImpact) -> str:
    parts = []
    out_ids = {id(p) for p in side.outgoing}
    for p, displaced, gain in side.upgrades:
        if displaced is not None and id(displaced) in out_ids:
            parts.append(f"{p.name} ({p.base:.1f} PPG) takes the {p.position} spot {displaced.name} ({displaced.base:.1f}) leaves, {gain:+.1f} PPG")
            continue
        over = (
            f"over {displaced.name} ({displaced.base:.1f})"
            if displaced else f"in a slot where {side.slot_notes.get(id(p), 'a waiver pickup would otherwise play')}"
        )
        parts.append(f"{p.name} ({p.base:.1f} PPG) starts at {p.position} {over}, +{gain:.1f} PPG")
    return "; ".join(parts)


PENALTY_LABELS = {"value lost": "value lost", "package hassle": "package hassle", "sideways swap": "sideways swap"}


def breakdown(side: SideImpact, is_target: bool) -> List[Tuple[str, str]]:
    """The trade quality breakdown for one side, as (label, text) rows"""
    rows = [
        ("Lineup", f"{side.lineup_delta:+.1f} PPG"),
        ("Value in / out", f"{side.value_in:.1f} / {side.value_out:.1f}"
                           + (f" (×{ENDOWMENT:.2f} endowment)" if is_target else "")
                           + f" → {VALUE_WEIGHT * side.value_delta:+.1f}"),
        ("Shape", side.shape),
        ("Need solved", ", ".join(side.need_solved) or "none"),
    ]
    if side.best_out is not None:
        rows.append(("Best player out", f"{side.best_out.name} ({side.best_out.base:.1f})"))
    if side.redundant:
        rows.append(("Redundant", _names(side.redundant)))
    if side.drops:
        rows.append(("Would cut", _names(side.drops)))
    if side.weakened:
        rows.append(("Weak spot thinned", f"{weakened_text(side)} → −{side.weakness:.1f}"))
    if is_target and (side.hassle or side.freed):
        rows.append(("Package", f"hassle −{side.hassle:.1f}" + (f", frees a roster spot +{side.freed:.1f}" if side.freed else "")))
    if is_target and side.lateral:
        rows.append(("Sideways swap", f"−{side.lateral:.1f} (same positions both ways, little lineup change)"))
    if side.risks:
        rows.append(("Risk", "; ".join(side.risks)))
    rows.append(("Total", f"{side.total:+.1f}"))
    return rows


def explain(t: TradeProposal) -> Tuple[str, str]:
    """(why it works for you, why it works for them)"""
    you = []
    if t.mine.upgrades:
        you.append(_upgrade_text(t.mine))
    else:
        you.append(f"your lineup changes by {t.mine.lineup_delta:+.1f} PPG")
    if t.mine.lost_starters:
        you.append(f"you give up {_names(t.mine.lost_starters)} from your lineup")
    if t.mine.value_delta < 0 and t.mine.best_out is not None:
        you.append(f"caution: {t.mine.best_out.name} is the best player in the deal, you net {VALUE_WEIGHT * t.mine.value_delta:+.1f} in value")
    elif t.mine.value_delta > 0:
        you.append(f"you also come out ahead on value ({VALUE_WEIGHT * t.mine.value_delta:+.1f})")
    if t.mine.redundant:
        you.append(f"note: {_names(t.mine.redundant)} would mostly sit on your bench")
    if t.mine.drops:
        you.append(f"you'd cut {_names(t.mine.drops)} to make room")
    if t.mine.weakened:
        you.append(f"caution: it thins your already-weak {weakened_text(t.mine)} (−{t.mine.weakness:.1f})")

    them = []
    if t.theirs.upgrades:
        them.append(_upgrade_text(t.theirs))
    else:
        them.append(f"their lineup changes by {t.theirs.lineup_delta:+.1f} PPG")
    them.append(f"for them it's a {t.theirs.shape}")
    if t.theirs.value_delta >= 0:
        them.append(f"they come out ahead on value ({VALUE_WEIGHT * t.theirs.value_delta:+.1f})")
    elif t.theirs.best_out is not None:
        role = "their backup " if t.theirs.best_out_share < 0.5 else "the best player in the deal, "
        them.append(f"they give up {role}{t.theirs.best_out.name}, value {VALUE_WEIGHT * t.theirs.value_delta:+.1f}")
    if t.theirs.hassle:
        them.append(f"package hassle −{t.theirs.hassle:.1f}")
    if t.theirs.lateral:
        them.append(f"sideways swap −{t.theirs.lateral:.1f}")
    if t.theirs.redundant:
        them.append(f"{_names(t.theirs.redundant)} would sit on their bench")
    if t.theirs.weakened:
        them.append(f"it thins their already-weak {weakened_text(t.theirs)} (−{t.theirs.weakness:.1f})")
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
    can't move the number. Returns (drv, live, sources): positions whose lookup fails fall back to
    FALLBACK_DRV, live is False unless every position came from real free agents, and sources
    lists the free agents behind each level so the number can be checked by eye.
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
