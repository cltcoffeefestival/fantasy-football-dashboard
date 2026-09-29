"""
Trade Finder - finds trades that improve your lineup and are plausible for the other team.

How it works
1. Each player gets a weekly value for the next N weeks:
   blend of season average and projection, x opponent matchup, x injury availability,
   and 0 on bye weeks.
2. Each team's best lineup is built from the league's real lineup slots (incl. flex). Any slot
   a rostered player can't beat (bye, injury, empty) is filled by a waiver-wire baseline, so
   bench players only count for what they add over a free pickup.
3. Candidate swaps with every other team are scored by the net change in weekly starting
   points (optimal lineup before vs. after) for BOTH teams. A trade is only suggested when it
   helps you meaningfully, helps them at least a little, and is roughly even in value.
4. Each player also gets an effective value (get_effective_value): starters count at their full
   projected average; bench players only for the bye weeks they cover, amortized over the season.
"""
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

WEEKS_AHEAD = 4

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

# Trade filters
MIN_MY_GAIN = 1.0        # pts/week
MIN_THEIR_GAIN = 0.5     # pts/week: otherwise they have no reason to accept
MIN_GAIN_BALANCE = 0.35  # their lineup gain must be at least this share of yours
MIN_VALUE_RATIO = 0.85   # trade value given vs. received must be within ~15%
SHOT_VALUE_RATIO = 0.7   # "worth a shot" trades can be a bit further apart
VALUE_EXPONENT = 1.5     # stars are worth more than the sum of two mid players
STAR_PREMIUM = 1.15      # extra weight on the single most valuable player on each side
TRADE_INJURY_DISCOUNT = {"DOUBTFUL": 0.95, "OUT": 0.7, "SUSPENSION": 0.8, "INJURY_RESERVE": 0.5}
# Creative mode loosens the partner-side filters so longer shots and bigger packages show up
CREATIVE_MIN_THEIR_GAIN = -1.5
CREATIVE_MIN_VALUE_RATIO = 0.4
FALLBACK_MIN_THEIR_GAIN = -3.0
FALLBACK_MIN_VALUE_RATIO = 0.3
CREATIVE_COMBOS = [(2, 2), (3, 1), (1, 3)]
POOL_SIZE = 8            # top players per side considered in a swap
STRENGTH_MARGIN = 0.10   # +/-10% vs. league average starters => strength / weakness

# Estimated waiver-wire replacement level (pts/wk) for a standard league. QB/RB/WR/TE are the
# agreed baselines; D/ST and K are my additions. Tune these if your scoring differs.
WAIVER_BASELINES = {"QB": 12.5, "RB": 7.0, "WR": 8.0, "TE": 5.5, "D/ST": 6.0, "K": 7.0}
SEASON_LAST_WEEK = 18


@dataclass
class PlayerValue:
    name: str
    position: str
    pro_team: str
    injury: str
    base: float                # blended points per game
    weekly: List[float] = field(default_factory=list)
    bye_weeks: List[int] = field(default_factory=list)   # remaining bye weeks this season

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


TIER_ORDER = ["Win-win", "Worth a shot", "Long shot"]


@dataclass
class TradeProposal:
    partner_id: int
    partner_name: str
    give: List[PlayerValue]
    get: List[PlayerValue]
    my_gain: float
    their_gain: float
    notes: List[str]
    tier: str = "Win-win"
    give_value: float = 0.0
    get_value: float = 0.0
    # Net weekly lineup impact: optimal starting points per week before and after the trade
    my_before: float = 0.0
    my_after: float = 0.0
    their_before: float = 0.0
    their_after: float = 0.0
    # (role, effective value) per player: my side's players in my lineup, theirs in theirs
    give_info: List[Tuple[str, float]] = field(default_factory=list)
    get_info: List[Tuple[str, float]] = field(default_factory=list)

    @property
    def rank_key(self) -> tuple:
        """Likelier trades first, then biggest combined gain"""
        return (TIER_ORDER.index(self.tier), -self.score)

    @property
    def score(self) -> float:
        return self.my_gain + self.their_gain


@dataclass
class LeagueSnapshot:
    teams: Dict[int, TeamSnapshot]
    slots: List[Tuple[str, frozenset]]
    weeks: List[int]
    schedule_adjusted: bool
    season_weeks_remaining: int = 1
    analysis: Dict[int, Dict[int, Tuple[str, float]]] = field(default_factory=dict)


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


def remaining_byes(schedule: dict, current_week: int, last_week: int = SEASON_LAST_WEEK) -> List[int]:
    """Weeks from now to season end with no game. Only trusts weeks the schedule actually covers."""
    if not schedule:
        return []
    covered = max(int(w) for w in schedule)
    return [
        w for w in range(current_week, min(last_week, covered) + 1)
        if not (schedule.get(str(w)) or schedule.get(w))
    ]


def build_player_value(
    player,
    weeks: List[int],
    ratings: Dict[int, Dict[str, Dict[str, int]]],
    current_week: Optional[int] = None,
) -> PlayerValue:
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
        bye_weeks=remaining_byes(schedule, current_week or (weeks[0] if weeks else 1)),
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


@lru_cache(maxsize=None)
def _slot_waiver(eligible: frozenset) -> float:
    """What a waiver pickup is worth in a slot: the best baseline among eligible positions"""
    return max((WAIVER_BASELINES.get(pos, 0.0) for pos in eligible), default=0.0)


def _solve_week(players: List[PlayerValue], slots, week: int):
    """Best lineup for one week. Returns (points, players who start).

    A slot goes to the best available rostered player, unless a waiver pickup would score more
    (bye, injury, or an empty slot), in which case the slot is filled at the waiver baseline.
    """
    ranked = sorted(players, key=lambda p: p.weekly[week], reverse=True)
    used = set()
    total = 0.0
    starters = []
    for _, eligible in slots:
        waiver = _slot_waiver(eligible)
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


def lineup_total(players: List[PlayerValue], slots, n_weeks: int) -> float:
    """Sum of the best possible lineup for each upcoming week"""
    return sum(_solve_week(players, slots, w)[0] for w in range(n_weeks))


def weeks_started(players: List[PlayerValue], slots, n_weeks: int) -> Dict[int, int]:
    """How many of the upcoming weeks each player is in the optimal lineup"""
    counts: Dict[int, int] = {}
    for w in range(n_weeks):
        for p in _solve_week(players, slots, w)[1]:
            counts[id(p)] = counts.get(id(p), 0) + 1
    return counts


def analyze_team(team: TeamSnapshot, slots, n_weeks: int, season_weeks_remaining: int) -> Dict[int, Tuple[str, float]]:
    """Role and effective weekly value of every player on a team, before any trade.

    Starter (in the optimal lineup at least half of the window): full projected average.
    Bench: only what he adds over a waiver pickup while covering the remaining bye weeks of the
    starters at his position, amortized over the remaining season:
        max(0, avg - waiver baseline) * bye weeks covered / weeks remaining
    Only the best backup at each position gets that credit.
    """
    started = weeks_started(team.players, slots, n_weeks)
    starters = [p for p in team.players if started.get(id(p), 0) * 2 >= n_weeks and started.get(id(p), 0) > 0]
    starter_ids = {id(p) for p in starters}

    result: Dict[int, Tuple[str, float]] = {id(p): ("Starter", p.base) for p in starters}

    covered_byes: Dict[str, set] = {}
    for p in starters:
        covered_byes.setdefault(p.position, set()).update(p.bye_weeks)

    bench: Dict[str, List[PlayerValue]] = {}
    for p in team.players:
        if id(p) not in starter_ids:
            bench.setdefault(p.position, []).append(p)
    for pos, players in bench.items():
        players.sort(key=lambda p: p.base, reverse=True)
        for rank, p in enumerate(players):
            value = 0.0
            if rank == 0:
                weeks_covered = len(covered_byes.get(pos, set()) - set(p.bye_weeks))
                upside = max(0.0, p.base - WAIVER_BASELINES.get(pos, 0.0))
                value = upside * weeks_covered / max(1, season_weeks_remaining)
            result[id(p)] = ("Bench", value)
    return result


def team_analysis(snapshot: "LeagueSnapshot", team: TeamSnapshot) -> Dict[int, Tuple[str, float]]:
    if team.team_id not in snapshot.analysis:
        snapshot.analysis[team.team_id] = analyze_team(
            team, snapshot.slots, len(snapshot.weeks), snapshot.season_weeks_remaining
        )
    return snapshot.analysis[team.team_id]


def get_effective_value(player: PlayerValue, team: TeamSnapshot, snapshot: "LeagueSnapshot") -> float:
    """Weekly value of a player to the team that owns him (starter vs. bench rules above)"""
    return team_analysis(snapshot, team).get(id(player), ("Bench", 0.0))[1]


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


# --------------------------------------------------------------------- trades

def _notes(give, get, partner_name, my_drop=(), their_drop=()) -> List[str]:
    notes = []
    for p in give + get:
        if p.injured:
            notes.append(f"{p.name} is {p.injury.replace('_', ' ').title()}")
        if p.weekly and p.weekly.count(0.0) and not p.injured:
            notes.append(f"{p.name} has a bye in the window")
    if my_drop:
        notes.append("You would drop " + ", ".join(p.name for p in my_drop))
    if their_drop:
        notes.append(f"{partner_name} would drop " + ", ".join(p.name for p in their_drop))
    return notes


def side_value(players, tv: Dict[int, float]) -> float:
    """Total trade value of a group of players, with a premium on the best one"""
    values = [tv[id(p)] for p in players]
    return sum(values) + (STAR_PREMIUM - 1) * max(values, default=0.0)


def trade_values(snapshot: "LeagueSnapshot") -> Dict[int, float]:
    """How an owner would value each player in a trade: points over replacement, stars weighted up.

    Ignores byes and matchups (owners don't discount a player for a bye), and only discounts
    serious injuries. This is separate from the lineup value used to measure the gain.
    """
    levels = WAIVER_BASELINES
    values = {}
    for team in snapshot.teams.values():
        for p in team.players:
            over = max(0.0, p.base - levels.get(p.position, 0.0))
            values[id(p)] = (over ** VALUE_EXPONENT) * TRADE_INJURY_DISCOUNT.get(p.injury, 1.0)
    return values


def classify(my_gain: float, their_gain: float, give_value: float, get_value: float) -> str:
    """How likely the other team is to say yes"""
    top = max(give_value, get_value)
    ratio = min(give_value, get_value) / top if top else 0
    if (
        their_gain >= MIN_THEIR_GAIN
        and their_gain >= MIN_GAIN_BALANCE * my_gain
        and ratio >= MIN_VALUE_RATIO
    ):
        return "Win-win"
    if (their_gain >= 0 and ratio >= SHOT_VALUE_RATIO) or (give_value >= get_value and their_gain >= 0):
        return "Worth a shot"
    return "Long shot"


def find_trades(
    snapshot: LeagueSnapshot,
    my_team_id: int,
    min_gain: float = MIN_MY_GAIN,
    max_results: Optional[int] = None,
    per_partner: int = 3,
    creative: bool = False,
    focus_positions: Optional[List[str]] = None,
) -> List[TradeProposal]:
    """Rank trades that help me, help them, and are roughly even in value.

    creative=True also returns longer shots (the other team gains little or loses a bit of
    lineup value, or you overpay in raw value) and bigger packages (2-for-2, 3-for-1, 1-for-3).
    focus_positions keeps only trades that bring back a player at one of those positions.
    """
    min_their = CREATIVE_MIN_THEIR_GAIN if creative else MIN_THEIR_GAIN
    min_ratio = CREATIVE_MIN_VALUE_RATIO if creative else MIN_VALUE_RATIO
    combos = [(1, 1), (2, 1), (1, 2)] + (CREATIVE_COMBOS if creative else [])
    focus = set(focus_positions or [])
    tv = trade_values(snapshot)
    n = len(snapshot.weeks)
    slots = snapshot.slots
    me = snapshot.teams.get(my_team_id)
    if not me or n == 0:
        return []

    def useful(players):
        return sorted((p for p in players if p.avg > 0), key=lambda p: p.avg, reverse=True)[:POOL_SIZE]

    def by_value(players):
        return sorted(players, key=lambda p: (tv[id(p)], p.avg))

    my_pool = useful(me.players)
    my_cheapest = by_value(me.players)
    my_before = lineup_total(me.players, slots, n)
    my_roles = team_analysis(snapshot, me)
    proposals: List[TradeProposal] = []

    def search_partner(tid, other, min_gain, min_their, min_ratio) -> List[TradeProposal]:
        their_pool = useful(other.players)
        their_before = lineup_total(other.players, slots, n)
        their_cheapest = by_value(other.players)
        their_roles = team_analysis(snapshot, other)
        found: List[TradeProposal] = []

        # 1-for-1 gains, kept for every pair so multi-player swaps can be compared to them
        single_gain: Dict[Tuple[int, int], float] = {}

        for give_n, get_n in combos:
            for give in combinations(my_pool, give_n):
                for get in combinations(their_pool, get_n):
                    give_ids = {id(p) for p in give}
                    get_ids = {id(p) for p in get}

                    # whoever ends up with extra players must cut the cheapest ones
                    my_drop = [p for p in my_cheapest if id(p) not in give_ids][: max(0, get_n - give_n)]
                    their_drop = [p for p in their_cheapest if id(p) not in get_ids][: max(0, give_n - get_n)]
                    give_value = max(0.0, side_value(give, tv) - sum(tv[id(p)] for p in their_drop))
                    get_value = max(0.0, side_value(get, tv) - sum(tv[id(p)] for p in my_drop))
                    if not max(give_value, get_value) or (
                        min(give_value, get_value) / max(give_value, get_value) < min_ratio
                    ):
                        continue

                    my_drop_ids = {id(p) for p in my_drop}
                    my_after = [
                        p for p in me.players if id(p) not in give_ids and id(p) not in my_drop_ids
                    ] + list(get)
                    my_after_total = lineup_total(my_after, slots, n)
                    my_gain = (my_after_total - my_before) / n

                    if (give_n, get_n) == (1, 1):
                        single_gain[(id(give[0]), id(get[0]))] = my_gain
                    else:
                        # skip swaps where the extra player adds nothing over a simpler 1-for-1
                        simpler = max(
                            single_gain.get((id(g), id(r)), float("-inf"))
                            for g in give for r in get
                        )
                        if simpler >= my_gain - 0.25:
                            continue
                    if my_gain < min_gain:
                        continue
                    if focus and not any(p.position in focus for p in get):
                        continue

                    their_drop_ids = {id(p) for p in their_drop}
                    their_after = [
                        p for p in other.players if id(p) not in get_ids and id(p) not in their_drop_ids
                    ] + list(give)
                    their_after_total = lineup_total(their_after, slots, n)
                    their_gain = (their_after_total - their_before) / n
                    if their_gain < min_their:
                        continue

                    found.append(TradeProposal(
                        partner_id=tid,
                        partner_name=other.name,
                        give=list(give),
                        get=list(get),
                        my_gain=my_gain,
                        their_gain=their_gain,
                        notes=_notes(list(give), list(get), other.name, my_drop, their_drop),
                        tier=classify(my_gain, their_gain, give_value, get_value),
                        give_value=give_value,
                        get_value=get_value,
                        my_before=my_before / n,
                        my_after=my_after_total / n,
                        their_before=their_before / n,
                        their_after=their_after_total / n,
                        give_info=[my_roles.get(id(p), ("Bench", 0.0)) for p in give],
                        get_info=[their_roles.get(id(p), ("Bench", 0.0)) for p in get],
                    ))

        found.sort(key=lambda t: t.rank_key)
        # keep the best few per partner, without repeating the same player on either side
        used_give, used_get = set(), set()
        kept_trades: List[TradeProposal] = []
        kept = 0
        for t in found:
            gids, rids = {id(p) for p in t.give}, {id(p) for p in t.get}
            if gids & used_give or rids & used_get:
                continue
            kept_trades.append(t)
            used_give |= gids
            used_get |= rids
            kept += 1
            if kept >= per_partner:
                break
        return kept_trades

    # Every team gets a look. In creative mode a team with no fit under the normal bar is
    # retried with a much lower bar, so you always see the best available angle with each team.
    attempts = [(min_gain, min_their, min_ratio)]
    if creative:
        attempts.append((min(min_gain, 0.25), FALLBACK_MIN_THEIR_GAIN, FALLBACK_MIN_VALUE_RATIO))
    for tid, other in snapshot.teams.items():
        if tid == my_team_id:
            continue
        for gain_bar, their_bar, ratio_bar in attempts:
            kept_trades = search_partner(tid, other, gain_bar, their_bar, ratio_bar)
            if kept_trades:
                proposals.extend(kept_trades)
                break

    proposals.sort(key=lambda t: t.rank_key)
    return proposals[:max_results] if max_results else proposals


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


def build_league_snapshot(league, weeks_ahead: int = WEEKS_AHEAD) -> LeagueSnapshot:
    """Read an espn_api League into plain data the engine can work on"""
    current = getattr(league, "current_week", None) or getattr(league, "nfl_week", 1)
    last_week = 18
    weeks = list(range(current, min(current + weeks_ahead, last_week + 1)))
    ratings = _fetch_ratings(league, weeks)

    teams = {}
    for team in league.teams:
        teams[team.team_id] = TeamSnapshot(
            team_id=team.team_id,
            name=team.team_name,
            players=[build_player_value(p, weeks, ratings, current) for p in team.roster],
            record=f"{team.wins}-{team.losses}",
        )

    return LeagueSnapshot(
        teams=teams,
        slots=build_slots(league.settings.position_slot_counts),
        weeks=weeks,
        schedule_adjusted=bool(ratings),
        season_weeks_remaining=max(1, last_week - current + 1),
    )
