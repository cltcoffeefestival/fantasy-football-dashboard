"""
Fantasy Football Dashboard - Professional UI
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from league_manager import LeagueManager
from analyzer import TeamAnalyzer
from trade_finder import (
    TIERS, WEEKS_AHEAD, build_league_snapshot, displaced_players, evaluate_custom, generate_trades, needs_table,
    DEFAULT_MIN_GAIN, FUNNEL_LABELS, asset_value, breakdown, positions_fixed, why_no_tier,
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
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


load_css()

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
def cached_trades(_lm, league_name, weeks_ahead, my_id, target_id, strict, min_gain):
    """Trade proposals for one target team (the snapshot underneath is cached too)"""
    snapshot = cached_snapshot(_lm, league_name, weeks_ahead)
    return generate_trades(snapshot, my_id, target_id, strict=strict, min_gain=min_gain)


TIER_STYLE = {
    "Win-Win": ("🟢", "WIN-WIN PROPOSALS (High Probability)", "Why They Accept"),
    "Worth a Shot": ("🟡", "WORTH A SHOT PROPOSALS (Medium Probability)", "Why They Might Accept / Hesitate"),
    "Long Shot": ("🔴", "LONG SHOT PROPOSALS (Aggressive / High-Reward)", "Why It's a Long Shot & How to Pitch It"),
}


def md_escape(text):
    """Team names can contain markdown characters (**Yamakas **); show them literally"""
    return re.sub(r"([*_`~\[\]])", r"\\\1", str(text))


def player_values_table(team, levels):
    """What the model sees for each player: PPG, window average, asset value, injury, weeks at zero"""
    rows = []
    for p in sorted(team.players, key=lambda p: (p.position, -p.base)):
        rows.append({
            "Player": p.name,
            "Pos": p.position,
            "PPG": round(p.base, 1),
            "Window avg": round(p.avg, 1),
            "Above last starter": round(p.base - levels.get(p.position, 0.0), 1),
            "Asset value": round(asset_value(p, levels), 1),
            "Injury": "" if not p.injured else p.injury.replace("_", " ").title(),
            "Zero weeks": sum(1 for w in p.weekly if w == 0),
        })
    return rows


def names(players):
    return " + ".join(f"**{p.name}** ({p.position}, {p.base:.1f} PPG)" for p in players)


def lineup_table(title, impact):
    """Slot-by-slot lineup before and after a trade, so the model's math can be checked by eye"""
    rows = []
    for (slot, before_who, before), (_, after_who, after) in zip(impact.lineup_before, impact.lineup_after):
        rows.append({
            "Slot": slot,
            "Before": f"{before_who} ({before:.1f})",
            "After": f"{after_who} ({after:.1f})",
            "Change": f"{after - before:+.1f}",
        })
    st.markdown(f"**{md_escape(title)}** · net {impact.delta:+.1f} PPG")
    st.dataframe(rows, width="stretch", hide_index=True)


def render_proposal(t, why_label=None):
    with st.container(border=True):
        st.markdown(f"**Proposed Trade:** You send {names(t.send)} ↔ You receive {names(t.receive)}")

        fixed = ", ".join(positions_fixed(t.mine)) or "none"
        displaced = ", ".join(f"{p.name} ({p.base:.1f})" for p in displaced_players(t.mine)) or "none"
        st.markdown(
            "**Impact Breakdown**\n"
            f"- **You:** lineup **{t.my_gain:+.1f} PPG** · value {t.mine.value_delta * 0.5:+.1f} · "
            f"worth **{t.my_effective_gain:+.1f}** to you · positions fixed: {fixed} · displaced: {displaced}\n"
            f"- **{md_escape(t.target_name)}:** lineup {t.delta_lineup:+.1f} PPG · value {t.theirs.value_delta * 0.5:+.1f}"
            + (f" · hassle −{t.theirs.hassle:.1f}" if t.theirs.hassle else "")
            + f" · acceptance **{t.acceptance:+.1f}**\n"
            f"- **Mutual:** {t.nmu:+.1f} · data confidence: {t.confidence}\n"
            f"- **Works because:** {md_escape(t.works_because)}\n"
            f"- **Might fail because:** {md_escape(t.fails_because)}"
        )
        st.markdown(f"**Roster Fit Summary**\n- **Why it works for you:** {md_escape(t.you_why)}\n- **Why it works for them:** {md_escape(t.them_why)}")
        with st.expander("Trade quality breakdown"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**You**")
                st.dataframe([{"": k, "Value": v} for k, v in breakdown(t.mine, False)], width="stretch", hide_index=True)
            with c2:
                st.markdown(f"**{md_escape(t.target_name)}**")
                st.dataframe([{"": k, "Value": v} for k, v in breakdown(t.theirs, True)], width="stretch", hide_index=True)
        with st.expander("Lineup before → after"):
            lineup_table("You", t.mine)
            lineup_table(t.target_name, t.theirs)
        for note in t.notes:
            st.caption(f"⚠️ {note}")
        if t.mine.drops:
            st.caption("You would cut: " + ", ".join(p.name for p in t.mine.drops))
        if t.theirs.drops:
            st.caption("They would cut: " + ", ".join(p.name for p in t.theirs.drops))


def render_misses(trades):
    """When few or no trades clear a tier: why, and the closest fair trades"""
    total = sum(len(trades[t]) for t in TIERS)
    if total == 0 and trades.funnel:
        parts = [f"{trades.funnel[k]} {FUNNEL_LABELS[k]}" for k in FUNNEL_LABELS if k not in ("qualified",) and trades.funnel.get(k)]
        st.caption("Nothing cleared a tier. " + "; ".join(parts) + ".")
    if trades.near_misses:
        with st.expander(f"Closest misses ({len(trades.near_misses)})", expanded=total == 0):
            st.caption("Plausible trades that don't clear a tier: fair or a stretch for them, and a small edge for you. Each says what it missed.")
            for t in trades.near_misses:
                st.markdown(f"⚪ {why_no_tier(t)}")
                render_proposal(t)


def render_tiers(trades):
    for tier in TIERS:
        icon, title, why_label = TIER_STYLE[tier]
        st.markdown(f"##### {icon} {title}")
        if not trades[tier]:
            st.caption("No proposal in this tier.")
        for t in trades[tier]:
            render_proposal(t)
    render_misses(trades)


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


def main():
    st.markdown(
        """
    <div class="app-header">
        <h1>Fantasy Football Dashboard</h1>
        <p>All your ESPN leagues in one place</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Initialize league manager
    league_manager = load_league_data()

    if not league_manager or not league_manager.leagues:
        st.error("❌ No leagues configured. Please add your league IDs to config.py")
        return

    analyzer = TeamAnalyzer(league_manager)

    # Sidebar navigation
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        [
            "🏠 Dashboard",
            "👥 My Teams",
            "📊 Waiver Wire",
            "🔁 Trade Finder",
            "🤝 Team Analysis",
            "🏆 Standings",
        ],
    )

    # Shared league / team picker (set once, used by every page)
    for name, err in league_manager.load_errors.items():
        st.sidebar.warning(f"Could not load **{name}**: {err}")

    st.sidebar.divider()
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
    if st.sidebar.button("🔄 Refresh data", width="stretch"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

    # Page: Dashboard / Overview
    if page == "🏠 Dashboard":
        all_teams = league_manager.get_user_teams()

        if all_teams.empty:
            st.warning("No team data available")
        else:
            only_mine = False
            if all_teams["Mine"].any():
                only_mine = st.toggle("Show only my teams", value=True)
                if only_mine:
                    all_teams = all_teams[all_teams["Mine"]]

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                with st.container(border=True):
                    st.metric("Leagues", all_teams["League"].nunique())
            with col2:
                with st.container(border=True):
                    st.metric("Teams", len(all_teams))
            with col3:
                with st.container(border=True):
                    st.metric("Total Wins", int(all_teams["Wins"].sum()))
            with col4:
                with st.container(border=True):
                    st.metric("Total Points For", f"{all_teams['Points For'].sum():,.0f}")

            CARDS_PER_ROW = 4
            for league in league_manager.leagues.keys():
                league_teams = all_teams[all_teams["League"] == league].sort_values("Standing")
                st.subheader(league)
                rows = [
                    league_teams.iloc[i : i + CARDS_PER_ROW]
                    for i in range(0, len(league_teams), CARDS_PER_ROW)
                ]
                for row in rows:
                    cols = st.columns(CARDS_PER_ROW)
                    for col, (_, team) in zip(cols, row.iterrows()):
                        with col:
                            render_team_card(team)

    # Page: My Teams
    elif page == "👥 My Teams":
        st.header("👥 My Teams")

        # Team Summary
        summary = analyzer.get_team_summary(league_name, team_id)
        if summary:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("📊 Record", summary["Record"])
            with col2:
                st.metric("🏅 Standing", f"#{summary['Standing']}")
            with col3:
                st.metric("📈 Points For", f"{summary['Points For']:.1f}")
            with col4:
                st.metric("⚡ PPW", f"{summary['PPW']:.1f}")

        st.markdown("---")

        # Roster
        st.subheader("📋 Current Roster")
        roster = league_manager.get_team_roster(league_name, team_id)

        if not roster.empty:
            # Group by position
            positions = ["QB", "RB", "WR", "TE", "D/ST", "K"]

            for position in positions:
                pos_roster = roster[roster['Position'] == position]
                if not pos_roster.empty:
                    with st.expander(f"**{position}** ({len(pos_roster)} players)", expanded=True):
                        st.dataframe(
                            pos_roster[["Name", "Team", "Avg Points", "Projected"]],
                            width="stretch",
                            hide_index=True,
                        )
        else:
            st.warning("No roster data available")

    # Page: Waiver Wire
    elif page == "📊 Waiver Wire":
        st.header("📋 Waiver Wire & Free Agents")

        limit = st.number_input("Free agents to consider:", value=50, min_value=10, step=10)

        # Recommended pickups
        st.subheader("🎯 Recommended Pickups by Position")
        recommendations = analyzer.recommend_waiver_pickups(league_name, team_id, pool_size=limit)
        if not recommendations.empty:
            cols = st.columns(3)
            for idx, (_, rec) in enumerate(recommendations.iterrows()):
                with cols[idx % 3]:
                    st.markdown(f"""
                    <div class="stat-box">
                        <h4>{rec['Position']}</h4>
                        <p style="font-size: 16px; font-weight: bold;">{rec['Player']}</p>
                        <p>Team: {rec['Team']}</p>
                        <p>Avg: {rec['Avg Points']:.1f} pts</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No waiver recommendations available")

        # All free agents
        st.markdown("---")
        st.subheader("📊 All Available Free Agents")
        free_agents = cached_free_agents(league_manager, league_name, limit)
        if not free_agents.empty:
            free_agents_sorted = free_agents.sort_values("Avg Points", ascending=False)

            # Filter by position
            position_filter = st.multiselect(
                "Filter by position:",
                free_agents_sorted['Position'].unique(),
                default=free_agents_sorted['Position'].unique()
            )

            filtered_agents = free_agents_sorted[free_agents_sorted['Position'].isin(position_filter)]

            st.dataframe(
                filtered_agents,
                width="stretch",
                hide_index=True,
            )
        else:
            st.warning("No free agents found")

    # Page: Trade Finder
    elif page == "🔁 Trade Finder":
        st.header("🔁 Trade Finder")
        st.caption(
            "A trade is only shown when it works for both managers. Each side is scored in PPG: the change to its "
            "best starting lineup over the window, plus half the asset value it nets (PPG above the league's last "
            "starter, with stars counting extra). The other manager is also charged for package hassle (3+ players) "
            "and sideways same-position swaps, and overvalues what he gives up a little. "
            "Win-Win: +1.5 or better for both in a one-for-one. Worth a Shot: they clear +1.0, you clear the slider. "
            "Long Shot: they're between −3.0 and +1.0, you gain +2.0 or more. Mutual = your worth + their acceptance."
        )

        c1, c2 = st.columns(2)
        with c1:
            weeks_ahead = st.slider("Weeks to look ahead", 1, 6, WEEKS_AHEAD)
        with c2:
            min_gain = st.slider(
                "Smallest gain for you (PPG)", 0.0, 2.0, DEFAULT_MIN_GAIN, 0.1,
                help="The smallest 'worth to you' (lineup change plus half the value you net) that still counts as "
                     "Worth a Shot. Lower it to see more trades.",
            )
            strict = st.toggle(
                "Strict acceptance (their score ≥ +1.5)",
                value=False,
                help="Hides trades the other manager only marginally clears. The Long Shot tier is empty in this mode.",
            )

        snapshot = cached_snapshot(league_manager, league_name, weeks_ahead)
        if team_id not in snapshot.teams:
            st.warning("Team not found in league data")
        else:
            st.caption(
                f"Weeks {snapshot.weeks[0]}–{snapshot.weeks[-1]} · "
                + ("replacement level from your league's actual free agents" if snapshot.drv_live
                   else "⚠️ free agent lookup incomplete, using estimated replacement levels")
                + " · "
                + ("matchup difficulty included" if snapshot.schedule_adjusted
                   else "⚠️ matchup ratings unavailable")
            )
            st.caption(
                "Last-starter levels (value is measured against these): "
                + ", ".join(f"{pos} {value:.1f}" for pos, value in snapshot.levels.items())
                + " · waiver levels (fill empty slots only): "
                + ", ".join(f"{pos} {value:.1f}" for pos, value in snapshot.drv.items())
            )
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
            others = {t.name: tid for tid, t in snapshot.teams.items() if tid != team_id}

            with st.expander("Your roster by position"):
                st.dataframe(needs_table(snapshot.teams, snapshot.slots, team_id), width="stretch", hide_index=True)
            with st.expander("Player values the model uses (yours)"):
                st.caption(
                    "PPG = blended season average and projection. Window avg = the next weeks after byes, "
                    "injuries and matchups. A player only helps a lineup where he beats the waiver level."
                )
                st.dataframe(
                    player_values_table(snapshot.teams[team_id], snapshot.levels), width="stretch", hide_index=True
                )

            target_choice = st.selectbox("Target team", ["Scan every team"] + list(others))
            if target_choice == "Scan every team":
                for name, tid in others.items():
                    trades = cached_trades(league_manager, league_name, weeks_ahead, team_id, tid, strict, min_gain)
                    total = sum(len(v) for v in trades.values())
                    best = next((TIER_STYLE[t][0] for t in TIERS if trades[t]), "⚪" if trades.near_misses else "·")
                    extra = f" + {len(trades.near_misses)} close" if not total and trades.near_misses else ""
                    label = f"{best} {md_escape(name)} ({snapshot.teams[tid].record}) · {total} proposal{'s' if total != 1 else ''}{extra}"
                    with st.expander(label, expanded=False):
                        render_tiers(trades)
            else:
                render_tiers(cached_trades(
                    league_manager, league_name, weeks_ahead, team_id, others[target_choice], strict, min_gain
                ))

            st.subheader("Score your own trade")
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
                icon = TIER_STYLE[t.tier][0] if t.tier else "⛔"
                st.markdown(f"{icon} **{t.tier or 'Does not clear any tier'}**")
                if not t.tier:
                    st.info(why_no_tier(t))
                render_proposal(t)

    # Page: Team Analysis
    elif page == "🤝 Team Analysis":
        st.header("🔍 Team Analysis")

        st.subheader("📊 Team Summary")
        summary = analyzer.get_team_summary(league_name, team_id)
        if summary:
            st.markdown(f"""
            <div class="stat-box">
                <p><strong>Team:</strong> {summary['Team']}</p>
                <p><strong>Record:</strong> {summary['Record']}</p>
                <p><strong>Points For:</strong> {summary['Points For']:.1f}</p>
                <p><strong>Points Against:</strong> {summary['Points Against']:.1f}</p>
                <p><strong>PPW:</strong> {summary['PPW']:.1f}</p>
                <p><strong>Roster Size:</strong> {summary['Roster Count']}</p>
            </div>
            """, unsafe_allow_html=True)

    # Page: League Standings
    elif page == "🏆 Standings":
        st.header("🏆 League Standings")

        standings = league_manager.get_league_standings(league_name)
        if not standings.empty:
            # Create visual standings
            fig = create_standings_visual(standings)
            if fig:
                st.plotly_chart(fig, width="stretch")

            st.markdown("---")

            # Table standings
            st.subheader("Detailed Standings")

            # Add rank column
            standings_display = standings.copy()
            standings_display.insert(0, 'Rank', range(1, len(standings_display) + 1))

            st.dataframe(
                standings_display,
                width="stretch",
                hide_index=True,
            )
        else:
            st.warning("No standings data available")

        # League info
        insights = analyzer.get_league_insights(league_name)
        if insights:
            st.markdown("---")
            st.subheader("League Information")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Teams", insights.get("Total Teams", "N/A"))
            with col2:
                st.metric("Current Week", insights.get("Current Week", "N/A"))


if __name__ == "__main__":
    main()
