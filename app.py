"""
Fantasy Football Dashboard - Professional UI
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from league_manager import LeagueManager
from analyzer import TeamAnalyzer
from config import LEAGUES
import logging
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

@st.cache_resource
def load_league_data():
    """Load league data with caching"""
    try:
        lm = LeagueManager(LEAGUES)
        return lm
    except Exception as e:
        st.error(f"Error loading leagues: {e}")
        return None


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
    with st.container(border=True):
        st.markdown(
            f'<div class="team-card-name">{html.escape(str(team["Team"]))}</div>'
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
            "🤝 Team Analysis",
            "🏆 Standings",
        ],
    )

    # Page: Dashboard / Overview
    if page == "🏠 Dashboard":
        all_teams = league_manager.get_user_teams()

        if all_teams.empty:
            st.warning("No team data available")
        else:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                with st.container(border=True):
                    st.metric("Leagues", len(league_manager.leagues))
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

        league_name = st.selectbox(
            "Select a league:",
            list(league_manager.leagues.keys()),
        )

        league = league_manager.leagues[league_name]
        team_options = {team.team_name: team.team_id for team in league.teams}

        selected_team = st.selectbox("Select your team:", list(team_options.keys()))
        team_id = team_options[selected_team]

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
            positions = ["QB", "RB", "WR", "TE", "DEF", "K"]

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

        league_name = st.selectbox(
            "Select a league:",
            list(league_manager.leagues.keys()),
            key="waiver_league",
        )

        league = league_manager.leagues[league_name]
        team_options = {team.team_name: team.team_id for team in league.teams}

        col1, col2 = st.columns([3, 1])
        with col1:
            selected_team = st.selectbox(
                "Select your team:", list(team_options.keys()), key="waiver_team"
            )
            team_id = team_options[selected_team]
        with col2:
            limit = st.number_input("Top N free agents:", value=50, min_value=10)

        # Recommended pickups
        st.subheader("🎯 Recommended Pickups by Position")
        recommendations = analyzer.recommend_waiver_pickups(league_name, team_id, top_n=6)
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
        free_agents = league_manager.get_free_agents(league_name, limit=limit)
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

    # Page: Team Analysis
    elif page == "🤝 Team Analysis":
        st.header("🔍 Team Analysis")

        league_name = st.selectbox(
            "Select a league:",
            list(league_manager.leagues.keys()),
            key="analysis_league",
        )

        league = league_manager.leagues[league_name]
        team_options = {team.team_name: team.team_id for team in league.teams}

        selected_team = st.selectbox(
            "Select your team:", list(team_options.keys()), key="analysis_team"
        )
        team_id = team_options[selected_team]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🤝 Potential Trade Partners")
            trades = analyzer.find_trade_opportunities(league_name, team_id)
            if trades.get("trade_partners"):
                partners_df = pd.DataFrame(trades["trade_partners"])

                # Color code by standing
                st.dataframe(
                    partners_df,
                    width="stretch",
                    hide_index=True,
                )

                st.info("💡 Trade partners are ranked by closeness in record")
            else:
                st.info("No close matches for trading right now")

        with col2:
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

        league_name = st.selectbox(
            "Select a league:",
            list(league_manager.leagues.keys()),
            key="standings_league",
        )

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
