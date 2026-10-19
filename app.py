"""
Fantasy Football Dashboard - Professional UI
"""
import streamlit as st
import plotly.graph_objects as go
from league_manager import LeagueManager
from analyzer import TeamAnalyzer
from trade_finder import (
    TIERS, WEEKS_AHEAD, build_league_snapshot, displaced_players, evaluate_custom, generate_trades, needs_table,
    DEFAULT_MIN_GAIN, FUNNEL_LABELS, apply_return_weeks, asset_value, breakdown, positions_fixed, why_no_tier,
)
from trade_history import (
    STATUS_LABELS, STATUSES, HistoryStore, espn_trades, fit_calibration, record_from_proposal,
    score_completed_trades, summarize,
)
from config import LEAGUES
from dataclasses import dataclass, field
import logging
import re
from pathlib import Path
import html

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Fantasy Football Dashboard",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css():
    """Inject the shared stylesheet"""
    css = (Path(__file__).parent / "assets" / "style.css").read_text()
    # st.html keeps the <style> tag; st.markdown strips it in current Streamlit, so the sheet never loaded
    st.html(f"<style>{css}</style>")


@st.cache_resource(ttl=900, show_spinner="Loading leagues from ESPN...")
def load_league_data():
    """Load league data with caching"""
    try:
        lm = LeagueManager(LEAGUES)
        return lm
    except Exception as e:
        st.error(f"Error loading leagues: {e}")
        return None


@st.cache_data(ttl=300, show_spinner=False)
def cached_free_agents(_lm, league_name, limit):
    """Free agent lookups hit ESPN every call, so cache them briefly"""
    return _lm.get_free_agents(league_name, limit=limit)


@st.cache_data(ttl=300, show_spinner="Scanning rosters, injuries and schedules...")
def cached_snapshot(_lm, league_name, weeks_ahead):
    """Player values for every roster in a league (one ESPN call per upcoming week)"""
    return build_league_snapshot(_lm.leagues[league_name], weeks_ahead)


@st.cache_data(ttl=300, show_spinner="Scanning rosters, free agents and schedules...")
def cached_trades(_lm, league_name, weeks_ahead, my_id, target_id, strict, min_gain, returns=(), calibration=None):
    """Trade proposals for one target team (the snapshot underneath is cached too). `returns` is a
    sorted tuple of (player name, NFL week he is expected back) overrides, part of the cache key, and
    `calibration` is what logged offers taught the engine (None to score untrained)."""
    snapshot = apply_return_weeks(cached_snapshot(_lm, league_name, weeks_ahead), dict(returns))
    return generate_trades(snapshot, my_id, target_id, strict=strict, min_gain=min_gain, calibration=calibration)


@st.cache_resource
def history_store():
    return HistoryStore()


@st.cache_data(ttl=120, show_spinner=False)
def cached_history(league_id, season):
    """Every logged offer and imported trade for one league season (empty when storage is off).
    Errors are raised, not cached, so a fixed key works on the next run."""
    store = history_store()
    if not store.enabled:
        return []
    return store.list(league_id, season)


@st.cache_data(ttl=300, show_spinner="Re-scoring the league's completed trades...")
def cached_completed_scores(_lm, league_name, trade_keys, _records):
    """Both sides' acceptance for every completed ESPN trade that can still be rebuilt. `trade_keys`
    (the trades' ESPN keys) is the cache key; scored on the default window so the result doesn't move
    with the Trade Finder's weeks slider."""
    return score_completed_trades(_records, cached_snapshot(_lm, league_name, WEEKS_AHEAD))


def completed_scores(league_manager, league_name, records):
    espn = [r for r in records if r.source == "espn"]
    if not espn:
        return []
    return cached_completed_scores(league_manager, league_name, tuple(sorted(r.espn_key or "" for r in espn)), espn)


def learning_label(records, completed):
    offers = sum(1 for r in records if r.source == "app" and r.status in ("accepted", "rejected", "countered")
                 and r.predicted_accept is not None)
    parts = []
    if offers:
        parts.append(f"{offers} answered offer{'s' if offers != 1 else ''}")
    if completed:
        parts.append(f"{len(completed)} completed trade{'s' if len(completed) != 1 else ''}")
    return " and ".join(parts)


def history_error(e):
    """A readable reason the trade log couldn't be reached"""
    status = getattr(getattr(e, "response", None), "status_code", None)
    if status in (401, 403):
        return ("Supabase rejected the key. Check that `SUPABASE_KEY` in the app's secrets is the project's "
                "secret key (starts with `sb_secret_`), pasted with nothing around it but quotes.")
    if status == 404:
        return "Supabase has no `trade_log` table at `SUPABASE_URL`. Check the URL, or create the table from `supabase/trade_log.sql`."
    return f"Couldn't reach the trade log: {e}"


@st.cache_data(ttl=900, show_spinner=False)
def import_league_trades(_league, league_id, season):
    """Copy completed ESPN trades into the log (at most every 15 minutes per league). A failed write
    raises, so it isn't cached and is retried on the next run."""
    store = history_store()
    if not store.enabled:
        return 0
    sent = store.import_espn(espn_trades(_league))
    cached_history.clear()
    return sent


@dataclass
class LogContext:
    """What a proposal card needs to log an offer"""
    store: HistoryStore
    league_id: int
    season: int
    week: int
    my_team: str
    logged: dict = field(default_factory=dict)     # (to_team, sends, receives) -> latest logged record


def trade_key(to_team, sends, receives):
    return (to_team, tuple(sorted(sends)), tuple(sorted(receives)))


PAGES = ["Overview", "My Team", "Waiver Wire", "Trade Finder", "Trade History", "Standings"]
PAGE_ICONS = {"Overview": "🏠", "My Team": "👥", "Waiver Wire": "📋", "Trade Finder": "🔁",
              "Trade History": "📜", "Standings": "🏆"}
POSITION_ORDER = ["QB", "RB", "WR", "TE", "D/ST", "K"]

# tier -> (icon, title, one-line meaning)
TIER_STYLE = {
    "Win-Win": ("🟢", "Win-Win", "A one-for-one that clearly helps both of you. Send it as is."),
    "Worth a Shot": ("🟡", "Worth a Shot", "Fair or slightly good for them, and a real gain for you. Expect a counter."),
    "Long Shot": ("🔴", "Long Shot", "A stretch for them but a big gain for you. Needs a pitch."),
}


def md_escape(text):
    """Team names can contain markdown characters (**Yamakas **); show them literally"""
    return re.sub(r"([*_`~\[\]])", r"\\\1", str(text))


def table(df, max_rows=20):
    """A dataframe tall enough to show its rows without an inner scrollbar (up to max_rows)"""
    st.dataframe(df, width="stretch", hide_index=True, height=35 * (min(len(df), max_rows) + 1) + 3)


def by_position(df, column="Position"):
    """Sort a roster-like table QB, RB, WR, TE, D/ST, K, then best first"""
    if df.empty:
        return df
    order = {pos: i for i, pos in enumerate(POSITION_ORDER)}
    df = df.copy()
    df["_pos"] = df[column].map(lambda p: order.get(p, len(order)))
    df = df.sort_values(["_pos", "Avg Points"], ascending=[True, False]).drop(columns="_pos")
    return df


# ------------------------------------------------------------------ trade finder widgets

def player_values_table(team, levels):
    """What the model sees for each player: PPG, window average, asset value, injury, weeks at zero"""
    rows = []
    for p in sorted(team.players, key=lambda p: (POSITION_ORDER.index(p.position) if p.position in POSITION_ORDER else 9, -p.base)):
        rows.append({
            "Player": p.name,
            "Pos": p.position,
            "PPG": round(p.base, 1),
            "Window avg": round(p.avg, 1),
            "Above last starter": round(p.base - levels.get(p.position, 0.0), 1),
            "Asset value": round(asset_value(p, levels), 1),
            "Injury": p.status_text if p.injured else "",
            "Zero weeks": sum(1 for w in p.weekly if w == 0),
        })
    return rows


def names(players):
    return " + ".join(f"**{p.name}** ({p.position}, {p.base:.1f} PPG)" for p in players)


def lineup_table(title, impact, weeks):
    """Who starts in each slot, week by week, before and after a trade, so the model's math can be
    checked by eye. A cell that the trade doesn't touch shows one name; a cell it changes shows
    'Lamar Jackson 20.9 → Waiver pickup 13.0'. The last column is the window average."""
    rows = []
    for before_row, after_row in zip(impact.lineup_before, impact.lineup_after):
        slot, _, before, before_weekly = before_row
        _, _, after, after_weekly = after_row
        row = {"Slot": slot}
        for i, ((b_name, b_pts), (a_name, a_pts)) in enumerate(zip(before_weekly, after_weekly)):
            label = f"Wk {weeks[i]}" if i < len(weeks) else f"Week {i + 1}"
            if b_name == a_name and abs(a_pts - b_pts) < 0.05:
                row[label] = f"{b_name} {b_pts:.1f}"
            else:
                row[label] = f"{b_name} {b_pts:.1f} → {a_name} {a_pts:.1f}"
        row["Avg"] = f"{before:.1f}" if abs(after - before) < 0.05 else f"{before:.1f} → {after:.1f} ({after - before:+.1f})"
        rows.append(row)
    st.markdown(f"**{md_escape(title)}** · net {impact.delta:+.1f} PPG")
    st.dataframe(rows, width="stretch", hide_index=True)


def render_log_control(t, log, scope):
    """Log this offer and its answer, so the engine can learn how this manager responds"""
    key = trade_key(t.target_name, [p.name for p in t.send], [p.name for p in t.receive])
    previous = log.logged.get(key)
    widget = "log_" + scope + "_" + "|".join([key[0], *key[1], "for", *key[2]])
    with st.popover("📝 Logged: " + STATUS_LABELS[previous.status] if previous else "📝 Log this offer"):
        if previous:
            st.caption(f"Logged {previous.created_at[:10] if previous.created_at else ''} as "
                       f"{STATUS_LABELS[previous.status].lower()}. Update it on the Trade History page.")
        status = st.radio("What happened?", STATUSES, format_func=STATUS_LABELS.get, key=widget + "_status")
        notes = st.text_input("Notes (optional)", key=widget + "_notes")
        if st.button("Save to history", key=widget + "_save", type="primary"):
            try:
                log.store.add(record_from_proposal(t, log.league_id, log.season, log.my_team, status,
                                                   week=log.week, notes=notes))
            except Exception as e:
                st.error(f"Could not save: {e}")
            else:
                cached_history.clear()
                st.toast("Saved to trade history")
                st.rerun()


def render_proposal(t, log=None, scope=""):
    """One trade card: the deal, the three numbers that decide it, the reasons, and the details folded away"""
    with st.container(border=True):
        st.markdown(f"You send {names(t.send)} &nbsp;↔&nbsp; you receive {names(t.receive)}")

        them = md_escape(t.target_name)
        c1, c2, c3 = st.columns(3)
        c1.metric("Worth to you", f"{t.my_effective_gain:+.1f}",
                  f"lineup {t.my_gain:+.1f} · value {t.mine.value_delta * 0.5:+.1f}", delta_color="off")
        c2.metric(f"Acceptance for {them}", f"{t.acceptance:+.1f}",
                  f"lineup {t.delta_lineup:+.1f} · value {t.theirs.value_delta * 0.5:+.1f}"
                  + (f" · hassle −{t.theirs.hassle:.1f}" if t.theirs.hassle else "")
                  + (f" · learned {t.learned_shift:+.1f}" if t.accept_chance is not None and abs(t.learned_shift) >= 0.05 else ""),
                  delta_color="off")
        c3.metric("Mutual", f"{t.nmu:+.1f}", f"data confidence {t.confidence}", delta_color="off")

        st.markdown(
            f"- **Works because:** {md_escape(t.works_because)}\n"
            f"- **Might fail because:** {md_escape(t.fails_because)}"
        )
        # a trade that clears no tier (only possible from "Score your own trade") is not a pitch, so the
        # roster-fit lines are headed neutrally instead of claiming it works
        you_label, them_label = ("Why it works for you", "Why it works for them") if t.tier else ("What it does to you", "What it does to them")
        st.markdown(f"- **{you_label}:** {md_escape(t.you_why)}\n- **{them_label}:** {md_escape(t.them_why)}")
        for note in t.notes:
            st.caption(f"⚠️ {note}")
        if t.mine.drops:
            st.caption("You would cut: " + ", ".join(p.name for p in t.mine.drops))
        if t.theirs.drops:
            st.caption("They would cut: " + ", ".join(p.name for p in t.theirs.drops))
        if t.accept_chance is not None:
            st.caption(f"Based on your logged offers, about a {t.accept_chance:.0%} chance {them} says yes.")
        if log is not None:
            render_log_control(t, log, scope)

        with st.expander("Details: scoring breakdown and lineups week by week"):
            fixed = ", ".join(positions_fixed(t.mine)) or "none"
            displaced = ", ".join(f"{p.name} ({p.base:.1f})" for p in displaced_players(t.mine)) or "none"
            st.caption(f"Positions fixed for you: {fixed} · your starters displaced: {displaced}")
            d1, d2 = st.columns(2)
            with d1:
                st.markdown("**You**")
                st.dataframe([{"": k, "Value": v} for k, v in breakdown(t.mine, False)], width="stretch", hide_index=True)
            with d2:
                st.markdown(f"**{them}**")
                st.dataframe([{"": k, "Value": v} for k, v in breakdown(t.theirs, True)], width="stretch", hide_index=True)
            st.caption("Who starts in each slot each week. A cell with an arrow is one the trade changes; "
                       "the rest stay as they are. Avg is the slot's average over the window.")
            lineup_table("You", t.mine, t.weeks)
            lineup_table(t.target_name, t.theirs, t.weeks)


def render_misses(trades, log=None):
    """When few or no trades clear a tier: why, and the closest fair trades"""
    total = sum(len(trades[t]) for t in TIERS)
    if total == 0 and trades.funnel:
        parts = [f"{trades.funnel[k]} {FUNNEL_LABELS[k]}" for k in FUNNEL_LABELS if k not in ("qualified",) and trades.funnel.get(k)]
        st.caption("Nothing cleared a tier. " + "; ".join(parts) + ".")
    if trades.near_misses:
        with st.expander(f"⚪ Closest misses ({len(trades.near_misses)})", expanded=total == 0):
            st.caption("Plausible trades that don't clear a tier: fair or a stretch for them, and a small edge for you. Each says what it missed.")
            for t in trades.near_misses:
                st.markdown(f"⚪ {why_no_tier(t)}")
                render_proposal(t, log, "miss")


def render_tiers(trades, log=None):
    total = sum(len(trades[t]) for t in TIERS)
    for tier in TIERS:
        if not trades[tier]:
            continue
        icon, title, meaning = TIER_STYLE[tier]
        st.markdown(f"#### {icon} {title}")
        st.caption(meaning)
        for t in trades[tier]:
            render_proposal(t, log, tier)
    if total:
        empty = [TIER_STYLE[t][1] for t in TIERS if not trades[t]]
        if empty:
            st.caption("No proposal in: " + ", ".join(empty) + ".")
    render_misses(trades, log)


# ------------------------------------------------------------------ overview widgets

def record_class(wins, losses):
    """CSS class for a record based on win percentage"""
    total = wins + losses
    if total == 0:
        return "rec-none"
    win_pct = wins / total
    if win_pct >= 0.6:
        return "rec-good"
    if win_pct >= 0.4:
        return "rec-mid"
    return "rec-bad"


def render_team_card(team):
    """One team card: rank, record, points"""
    wins, losses = int(team["Wins"]), int(team["Losses"])
    you_badge = '<span class="rank-badge">YOU</span>' if team.get("Mine") else ""
    with st.container(border=True):
        st.markdown(
            f'<div class="team-card-name">{html.escape(str(team["Team"]))}'
            f' {you_badge}</div>'
            f'<div class="team-card-record {record_class(wins, losses)}">{wins}-{losses}</div>'
            f'<div class="team-card-meta">'
            f'<span class="rank-badge">#{int(team["Standing"])}</span>'
            f'{team["Points For"]:.1f} PF · {team["Points Against"]:.1f} PA</div>',
            unsafe_allow_html=True,
        )


def create_standings_visual(standings_df):
    """Horizontal bar chart of points for, best team on top"""
    if standings_df.empty:
        return None

    df = standings_df.sort_values("Points For")
    fig = go.Figure(
        go.Bar(
            y=df["Team"],
            x=df["Points For"],
            orientation="h",
            marker=dict(color="#4f46e5"),
            text=df["Points For"],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>%{x} PF<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(300, 34 * len(df) + 80),
        showlegend=False,
        margin=dict(l=0, r=40, t=10, b=0),
        xaxis_title=None,
        yaxis_title=None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)")
    return fig


def page_title(page, subtitle):
    st.markdown(
        f'<div class="app-header"><h1>{PAGE_ICONS[page]} {html.escape(page)}</h1><p>{html.escape(subtitle)}</p></div>',
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------ pages

def overview_page(league_manager):
    page_title("Overview", "All your ESPN leagues in one place")
    all_teams = league_manager.get_user_teams()
    if all_teams.empty:
        st.warning("No team data available")
        return

    if all_teams["Mine"].any():
        if st.toggle("Show only my teams", value=True):
            all_teams = all_teams[all_teams["Mine"]]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Leagues", all_teams["League"].nunique())
    m2.metric("Teams", len(all_teams))
    m3.metric("Total wins", int(all_teams["Wins"].sum()))
    m4.metric("Total points for", f"{all_teams['Points For'].sum():,.0f}")

    CARDS_PER_ROW = 4
    for league in league_manager.leagues.keys():
        league_teams = all_teams[all_teams["League"] == league].sort_values("Standing")
        if league_teams.empty:
            continue
        st.subheader(league)
        for i in range(0, len(league_teams), CARDS_PER_ROW):
            cols = st.columns(CARDS_PER_ROW)
            for col, (_, team) in zip(cols, league_teams.iloc[i: i + CARDS_PER_ROW].iterrows()):
                with col:
                    render_team_card(team)


def my_team_page(league_manager, analyzer, league_name, team_id, selected_team):
    page_title("My Team", f"{selected_team} · {league_name}")
    summary = analyzer.get_team_summary(league_name, team_id)
    if summary:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Record", summary["Record"])
        m2.metric("Standing", f"#{summary['Standing']}")
        m3.metric("Points for", f"{summary['Points For']:.1f}")
        m4.metric("Points against", f"{summary['Points Against']:.1f}")
        m5.metric("Points per week", f"{summary['PPW']:.1f}")

    roster = league_manager.get_team_roster(league_name, team_id)
    if roster.empty:
        st.warning("No roster data available")
        return

    injuries = analyzer.get_injury_report(league_name)
    if not injuries.empty:
        mine = injuries[injuries["Player"].isin(roster["Name"])]
        if not mine.empty:
            st.warning("Injuries: " + ", ".join(f"{r.Player} ({r.Status.replace('_', ' ').title()})" for r in mine.itertuples()))

    st.subheader(f"Roster · {len(roster)} players")
    table(by_position(roster)[["Position", "Name", "Team", "Avg Points", "Projected"]])


def waiver_page(league_manager, analyzer, league_name, team_id):
    page_title("Waiver Wire", "Best available free agents in this league")
    c1, c2 = st.columns([1, 2])
    with c1:
        limit = st.number_input("Free agents to consider", value=50, min_value=10, step=10)
    free_agents = cached_free_agents(league_manager, league_name, limit)
    if free_agents.empty:
        st.warning("No free agents found")
        return
    with c2:
        positions = [p for p in POSITION_ORDER if p in set(free_agents["Position"])]
        position_filter = st.multiselect("Positions", positions, default=positions)

    st.subheader("Best available at each position")
    recommendations = analyzer.recommend_waiver_pickups(league_name, team_id, pool_size=limit)
    if recommendations.empty:
        st.info("No waiver recommendations available")
    else:
        cols = st.columns(3)
        for idx, (_, rec) in enumerate(recommendations.iterrows()):
            with cols[idx % 3]:
                with st.container(border=True):
                    st.markdown(
                        f'<div class="team-card-meta">{html.escape(str(rec["Position"]))} · {html.escape(str(rec["Team"]))}</div>'
                        f'<div class="team-card-name">{html.escape(str(rec["Player"]))}</div>'
                        f'<div class="team-card-meta">{rec["Avg Points"]:.1f} avg · {rec["Projected"]:.1f} projected</div>',
                        unsafe_allow_html=True,
                    )

    st.subheader("All free agents")
    shown = by_position(free_agents[free_agents["Position"].isin(position_filter)])
    st.dataframe(shown.sort_values("Avg Points", ascending=False), width="stretch", hide_index=True)


def standings_page(league_manager, analyzer, league_name):
    page_title("Standings", league_name)
    insights = analyzer.get_league_insights(league_name)
    if insights:
        st.caption(f"Week {insights.get('Current Week', '?')} · {insights.get('Total Teams', '?')} teams")
    standings = league_manager.get_league_standings(league_name)
    if standings.empty:
        st.warning("No standings data available")
        return
    fig = create_standings_visual(standings)
    if fig:
        st.plotly_chart(fig, width="stretch")
    standings_display = standings.drop(columns=["Standing"]).copy()
    standings_display.insert(0, "Rank", range(1, len(standings_display) + 1))
    table(standings_display)


def trade_finder_page(league_manager, league_name, team_id):
    page_title("Trade Finder", "Trades the other manager would actually accept")

    with st.container(border=True):
        c1, c2, c3 = st.columns([3, 3, 2])
        with c1:
            weeks_ahead = st.slider("Weeks to look ahead", 1, 6, WEEKS_AHEAD)
        with c2:
            min_gain = st.slider(
                "Smallest gain for you (PPG)", 0.0, 2.0, DEFAULT_MIN_GAIN, 0.1,
                help="The smallest 'worth to you' (lineup change plus half the value you net) that still counts as "
                     "Worth a Shot. Lower it to see more trades.",
            )
        with c3:
            strict = st.toggle(
                "Strict",
                value=False,
                help="Only trades the other manager clearly clears (their score ≥ +1.5). Long Shots are hidden.",
            )

        snapshot = cached_snapshot(league_manager, league_name, weeks_ahead)
        if team_id not in snapshot.teams:
            st.warning("Team not found in league data")
            return

        # ESPN's injury tag is run through a fixed curve (Out = mostly gone for the window). When you
        # know better, say which week a player is back and every number below is recomputed with it.
        sidelined = sorted(
            (p for t in snapshot.teams.values() for p in t.players
             if p.injury in ("OUT", "DOUBTFUL", "INJURY_RESERVE", "SUSPENSION")),
            key=lambda p: (p.injury != "OUT", -p.base),
        )
        returns = {}
        if sidelined and len(snapshot.weeks) > 1:
            with st.expander("Injured players you expect back sooner"):
                st.caption(
                    "An Out tag is modeled as 0%, 30%, 60% and 85% likely to play over the next four weeks "
                    "(IR and suspensions longer). Pick a player and the week he is back to override that; "
                    "he then sits until that week and plays every week after."
                )
                labels = {f"{p.name} ({p.position}, {p.injury.replace('_', ' ').title()})": p.name for p in sidelined}
                chosen = st.multiselect("Players", list(labels), key="returns_players")
                for label in chosen:
                    week = st.selectbox(
                        f"{labels[label]} is back in week", snapshot.weeks[1:], key=f"returns_{labels[label]}",
                    )
                    returns[labels[label]] = int(week)
        returns_key = tuple(sorted(returns.items()))
        snapshot = apply_return_weeks(snapshot, returns)

        # what logged offers taught the engine about this league's managers
        league = league_manager.leagues[league_name]
        store = history_store()
        log, calibration = None, None
        if store.enabled:
            try:
                records = cached_history(league.league_id, league.year)
            except Exception as e:
                st.warning(history_error(e))
                records = []
            log = LogContext(store, league.league_id, league.year, getattr(league, "current_week", None),
                             snapshot.teams[team_id].name)
            for r in sorted((r for r in records if r.source == "app"), key=lambda r: r.created_at or ""):
                log.logged[trade_key(r.to_team, r.sends, r.receives)] = r
            completed = completed_scores(league_manager, league_name, records)
            learned = fit_calibration(records, completed)
            if learned.trained:
                use = st.toggle(
                    f"Learn from {learning_label(records, completed)}",
                    value=True,
                    help="Adjusts every acceptance score by how this league's managers actually answered the "
                         "offers you logged, and by the trades they completed (those count half). See the "
                         "Trade History page for what it learned.",
                )
                calibration = learned if use else None
        st.caption(
            f"Weeks {snapshot.weeks[0]}–{snapshot.weeks[-1]} · "
            + ("replacement level from this league's free agents" if snapshot.drv_live
               else "⚠️ free agent lookup incomplete, using estimated replacement levels")
            + " · "
            + ("matchup difficulty included" if snapshot.schedule_adjusted else "⚠️ matchup ratings unavailable")
            + (" · return weeks applied: " + ", ".join(f"{name} wk {week}" for name, week in returns_key) if returns else "")
            + (" · acceptance learned from your logged offers" if calibration else "")
        )

    others = {t.name: tid for tid, t in snapshot.teams.items() if tid != team_id}
    find_tab, score_tab, model_tab = st.tabs(["Find trades", "Score your own trade", "How it scores"])

    with find_tab:
        target_choice = st.selectbox("Trade partner", ["Scan every team"] + list(others))
        if target_choice == "Scan every team":
            for name, tid in others.items():
                trades = cached_trades(league_manager, league_name, weeks_ahead, team_id, tid, strict, min_gain, returns_key,
                                       calibration)
                total = sum(len(v) for v in trades.values())
                best = next((TIER_STYLE[t][0] for t in TIERS if trades[t]), "⚪" if trades.near_misses else "⚫")
                extra = f" + {len(trades.near_misses)} close" if not total and trades.near_misses else ""
                label = f"{best} {md_escape(name)} ({snapshot.teams[tid].record}) · {total} proposal{'s' if total != 1 else ''}{extra}"
                with st.expander(label, expanded=False):
                    render_tiers(trades, log)
        else:
            render_tiers(cached_trades(
                league_manager, league_name, weeks_ahead, team_id, others[target_choice], strict, min_gain, returns_key,
                calibration,
            ), log)

    with score_tab:
        e1, e2 = st.columns(2)
        with e1:
            eval_target = st.selectbox("Trade with", list(others), key="eval_target")
            mine = snapshot.teams[team_id].players
            send_names = st.multiselect("You send", [p.name for p in mine], key="eval_send")
        with e2:
            theirs = snapshot.teams[others[eval_target]].players
            recv_names = st.multiselect("You receive", [p.name for p in theirs], key="eval_recv")
        if send_names and recv_names:
            t = evaluate_custom(
                snapshot, team_id, others[eval_target],
                [p for p in mine if p.name in send_names],
                [p for p in theirs if p.name in recv_names],
                min_gain=min_gain,
                calibration=calibration,
            )
            if t.tier:
                icon, title, meaning = TIER_STYLE[t.tier]
                st.markdown(f"#### {icon} {title}")
                st.caption(meaning)
            else:
                st.markdown("#### ⛔ Does not clear any tier")
                st.info(why_no_tier(t))
            render_proposal(t, log, "custom")
        else:
            st.caption("Pick what you send and what you receive to score the trade with the same math as the finder.")

    with model_tab:
        st.markdown(
            "Each side is scored in PPG: the change to its best starting lineup over the window, plus half the "
            "asset value it nets (PPG above the league's last starter, with stars counting extra). The other "
            "manager is also charged for package hassle (3+ players) and sideways same-position swaps, and "
            "overvalues what he gives up a little.\n\n"
            "- 🟢 **Win-Win**: +1.5 or better for both, one-for-one only.\n"
            "- 🟡 **Worth a Shot**: they clear +1.0, you clear the slider.\n"
            "- 🔴 **Long Shot**: they're between −3.0 and +1.0, you gain +2.0 or more.\n\n"
            "Mutual = your worth + their acceptance.\n\n"
            "Once you log offers and their answers (📝 on any trade), acceptance is also corrected by how "
            "this league's managers actually answered. The Trade History page shows what it learned."
        )
        st.caption(
            "Last-starter levels (value is measured against these): "
            + ", ".join(f"{pos} {value:.1f}" for pos, value in snapshot.levels.items())
            + " · waiver levels (fill empty slots only): "
            + ", ".join(f"{pos} {value:.1f}" for pos, value in snapshot.drv.items())
        )
        with st.expander("Your roster by position"):
            st.dataframe(needs_table(snapshot.teams, snapshot.slots, team_id), width="stretch", hide_index=True)
        with st.expander("Player values the model uses (yours)"):
            st.caption(
                "PPG = blended season average and projection. Window avg = the next weeks after byes, "
                "injuries and matchups. A player only helps a lineup where he beats the waiver level."
            )
            st.dataframe(player_values_table(snapshot.teams[team_id], snapshot.levels), width="stretch", hide_index=True)
        with st.expander("Free agents behind the waiver levels"):
            st.caption(
                "Each waiver level is the average of the best few healthy free agents at the position. "
                "It only fills a lineup slot nobody on the roster can play (bye, injury, no player there); "
                "player value is measured against the league's last starter, not the waiver wire."
            )
            rows = [
                {"Position": pos, "Free agent": name, "PPG used": round(blend, 1),
                 "Season avg": round(avg, 1), "Projection": round(proj, 1)}
                for pos, found in snapshot.drv_sources.items()
                for name, blend, avg, proj in found
            ]
            st.dataframe(rows, width="stretch", hide_index=True)


def trade_history_page(league_manager, league_name):
    page_title("Trade History", "Offers you logged, trades completed in the league, and what the model learned")
    store = history_store()
    if not store.enabled:
        st.info(
            "Trade history needs a database so it survives app restarts. Add `SUPABASE_URL` and "
            "`SUPABASE_KEY` (the project's secret key) to the app's secrets, then reload."
        )
        return
    league = league_manager.leagues[league_name]
    try:
        import_league_trades(league, league.league_id, league.year)
        records = cached_history(league.league_id, league.year)
    except Exception as e:
        st.error(history_error(e))
        return
    mine = [r for r in records if r.source == "app"]
    completed = [r for r in records if r.source == "espn"]
    completed_scored = completed_scores(league_manager, league_name, records)
    summary = summarize(records, completed_scored)
    cal = summary.calibration

    # ---- what the model learned
    with st.container(border=True):
        st.markdown("#### What the model learned")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Offers logged", len(mine), f"{summary.pending} awaiting an answer", delta_color="off")
        c2.metric("Accepted", summary.accepted, f"{summary.countered} countered · {summary.rejected} rejected", delta_color="off")
        if summary.answered:
            c3.metric("Predicted yes rate", f"{summary.predicted_rate:.0%}", "before learning", delta_color="off")
            c4.metric("Actual yes rate", f"{summary.actual_rate:.0%}", "a counter counts half", delta_color="off")
        else:
            c3.metric("Predicted yes rate", "–")
            c4.metric("Actual yes rate", "–")
        if not cal.trained:
            st.caption("Nothing to learn from yet. Log offers from the Trade Finder (📝 on any trade) and mark "
                       "how they were answered; every answered offer and completed league trade adjusts the "
                       "acceptance scores.")
        else:
            def said(shift):
                if abs(shift) < 0.05:
                    return "about as predicted"
                return f"{shift:+.1f} PPG {'easier' if shift > 0 else 'harder'} to please than the model assumed"
            st.markdown(
                f"- **One-for-one offers:** managers here are {said(summary.single_shift)}.\n"
                f"- **Multi-player offers:** {said(summary.multi_shift)}."
            )
            st.caption(
                f"Learned from {learning_label(records, completed_scored)}. With little data the model stays close "
                "to its defaults; each new answer moves it a little more. A completed trade is re-scored by putting "
                "the players back on their old rosters (with today's values), and counts as half an accepted offer, "
                "because ESPN only shows the trades that went through."
            )
        if summary.completed_total:
            line = (f"Completed trades used: {summary.completed_used} of {summary.completed_total}"
                    + (" (the rest have a player who has since moved on)" if summary.completed_used < summary.completed_total else ""))
            if summary.completed_used:
                line += f". The untrained model gave their sides a {summary.completed_predicted:.0%} chance on average; all of them said yes"
            st.caption(line + ".")

    # ---- your offers
    st.markdown("#### Your logged offers")
    if not mine:
        st.caption("No offers logged yet.")
    else:
        rows = [{
            "id": r.id,
            "Date": (r.created_at or "")[:10],
            "To": r.to_team,
            "You send": ", ".join(r.sends),
            "You receive": ", ".join(r.receives),
            "Predicted": None if r.predicted_accept is None else round(r.predicted_accept, 1),
            "Tier": r.tier or "",
            "Status": STATUS_LABELS[r.status],
            "Notes": r.notes,
        } for r in mine]
        edited = st.data_editor(
            rows, hide_index=True, width="stretch", key=f"history_{league.league_id}",
            disabled=["id", "Date", "To", "You send", "You receive", "Predicted", "Tier"],
            column_config={
                "id": None,
                "Status": st.column_config.SelectboxColumn(options=[STATUS_LABELS[s] for s in STATUSES], required=True),
                "Predicted": st.column_config.NumberColumn(help="The model's acceptance score for them when you logged it"),
            },
        )
        label_to_status = {v: k for k, v in STATUS_LABELS.items()}
        changes = [(new, old) for new, old in zip(edited, rows)
                   if new["Status"] != old["Status"] or (new["Notes"] or "") != (old["Notes"] or "")]
        if changes and st.button(f"Save {len(changes)} change{'s' if len(changes) != 1 else ''}", type="primary"):
            try:
                for new, _ in changes:
                    store.update(new["id"], status=label_to_status[new["Status"]], notes=new["Notes"] or None)
            except Exception as e:
                st.error(f"Could not save: {e}")
            else:
                cached_history.clear()
                st.toast("History updated")
                st.rerun()

    # ---- the league's completed trades
    st.markdown("#### Trades completed in the league")
    if not completed:
        st.caption("No completed trades found on ESPN this season.")
    else:
        table([{
            "Date": (r.created_at or "")[:10],
            "Team A": r.from_team,
            "A sends": ", ".join(r.sends),
            "Team B": r.to_team,
            "B sends": ", ".join(r.receives),
        } for r in completed])
    if summary.by_manager:
        with st.expander("By manager"):
            table(summary.by_manager)


def main():
    load_css()
    league_manager = load_league_data()
    if not league_manager or not league_manager.leagues:
        st.error("❌ No leagues configured. Please add your league IDs to config.py")
        return
    analyzer = TeamAnalyzer(league_manager)

    # Sidebar: the league and team every page works on, then the pages
    st.sidebar.markdown("### 🏈 Fantasy Football")
    for name, err in league_manager.load_errors.items():
        st.sidebar.warning(f"Could not load **{name}**: {err}")
    league_name = st.sidebar.selectbox("League", list(league_manager.leagues.keys()))
    league = league_manager.leagues[league_name]
    team_options = {team.team_name: team.team_id for team in league.teams}
    my_names = [t.team_name for t in league.teams if league_manager.is_my_team(t)]
    default_idx = list(team_options).index(my_names[0]) if my_names else 0
    selected_team = st.sidebar.selectbox(
        "Your team", list(team_options.keys()), index=default_idx, key=f"team_{league_name}"
    )
    team_id = team_options[selected_team]
    if not my_names:
        st.sidebar.caption("Couldn't match your SWID to a team; pick yours above.")

    st.sidebar.divider()
    page = st.sidebar.radio("Pages", PAGES, format_func=lambda p: f"{PAGE_ICONS[p]} {p}", label_visibility="collapsed")
    st.sidebar.divider()
    if st.sidebar.button("🔄 Refresh data", width="stretch"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

    if page == "Overview":
        overview_page(league_manager)
    elif page == "My Team":
        my_team_page(league_manager, analyzer, league_name, team_id, selected_team)
    elif page == "Waiver Wire":
        waiver_page(league_manager, analyzer, league_name, team_id)
    elif page == "Trade Finder":
        trade_finder_page(league_manager, league_name, team_id)
    elif page == "Trade History":
        trade_history_page(league_manager, league_name)
    elif page == "Standings":
        standings_page(league_manager, analyzer, league_name)


if __name__ == "__main__":
    main()
