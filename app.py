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
from config import LEAGUES
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
def cached_trades(_lm, league_name, weeks_ahead, my_id, target_id, strict, min_gain, returns=()):
    """Trade proposals for one target team (the snapshot underneath is cached too). `returns` is a
    sorted tuple of (player name, NFL week he is expected back) overrides, part of the cache key."""
    snapshot = apply_return_weeks(cached_snapshot(_lm, league_name, weeks_ahead), dict(returns))
    return generate_trades(snapshot, my_id, target_id, strict=strict, min_gain=min_gain)


PAGES = ["Overview", "My Team", "Waiver Wire", "Trade Finder", "Standings"]
PAGE_ICONS = {"Overview": "🏠", "My Team": "👥", "Waiver Wire": "📋", "Trade Finder": "🔁", "Standings": "🏆"}
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


def render_proposal(t):
    """One trade card: the deal, the three numbers that decide it, the reasons, and the details folded away"""
    with st.container(border=True):
        st.markdown(f"You send {names(t.send)} &nbsp;↔&nbsp; you receive {names(t.receive)}")

        them = md_escape(t.target_name)
        c1, c2, c3 = st.columns(3)
        c1.metric("Worth to you", f"{t.my_effective_gain:+.1f}",
                  f"lineup {t.my_gain:+.1f} · value {t.mine.value_delta * 0.5:+.1f}", delta_color="off")
        c2.metric(f"Acceptance for {them}", f"{t.acceptance:+.1f}",
                  f"lineup {t.delta_lineup:+.1f} · value {t.theirs.value_delta * 0.5:+.1f}"
                  + (f" · hassle −{t.theirs.hassle:.1f}" if t.theirs.hassle else ""), delta_color="off")
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


def render_misses(trades):
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
                render_proposal(t)


def render_tiers(trades):
    total = sum(len(trades[t]) for t in TIERS)
    for tier in TIERS:
        if not trades[tier]:
            continue
        icon, title, meaning = TIER_STYLE[tier]
        st.markdown(f"#### {icon} {title}")
        st.caption(meaning)
        for t in trades[tier]:
            render_proposal(t)
    if total:
        empty = [TIER_STYLE[t][1] for t in TIERS if not trades[t]]
        if empty:
            st.caption("No proposal in: " + ", ".join(empty) + ".")
    render_misses(trades)


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
        st.caption(
            f"Weeks {snapshot.weeks[0]}–{snapshot.weeks[-1]} · "
            + ("replacement level from this league's free agents" if snapshot.drv_live
               else "⚠️ free agent lookup incomplete, using estimated replacement levels")
            + " · "
            + ("matchup difficulty included" if snapshot.schedule_adjusted else "⚠️ matchup ratings unavailable")
            + (" · return weeks applied: " + ", ".join(f"{name} wk {week}" for name, week in returns_key) if returns else "")
        )

    others = {t.name: tid for tid, t in snapshot.teams.items() if tid != team_id}
    find_tab, score_tab, model_tab = st.tabs(["Find trades", "Score your own trade", "How it scores"])

    with find_tab:
        target_choice = st.selectbox("Trade partner", ["Scan every team"] + list(others))
        if target_choice == "Scan every team":
            for name, tid in others.items():
                trades = cached_trades(league_manager, league_name, weeks_ahead, team_id, tid, strict, min_gain, returns_key)
                total = sum(len(v) for v in trades.values())
                best = next((TIER_STYLE[t][0] for t in TIERS if trades[t]), "⚪" if trades.near_misses else "⚫")
                extra = f" + {len(trades.near_misses)} close" if not total and trades.near_misses else ""
                label = f"{best} {md_escape(name)} ({snapshot.teams[tid].record}) · {total} proposal{'s' if total != 1 else ''}{extra}"
                with st.expander(label, expanded=False):
                    render_tiers(trades)
        else:
            render_tiers(cached_trades(
                league_manager, league_name, weeks_ahead, team_id, others[target_choice], strict, min_gain, returns_key
            ))

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
            )
            if t.tier:
                icon, title, meaning = TIER_STYLE[t.tier]
                st.markdown(f"#### {icon} {title}")
                st.caption(meaning)
            else:
                st.markdown("#### ⛔ Does not clear any tier")
                st.info(why_no_tier(t))
            render_proposal(t)
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
            "Mutual = your worth + their acceptance."
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
    elif page == "Standings":
        standings_page(league_manager, analyzer, league_name)


if __name__ == "__main__":
    main()
