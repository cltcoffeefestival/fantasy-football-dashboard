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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Fantasy Football Dashboard",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling
st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .team-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .record-good {
        color: #00cc44;
        font-weight: bold;
    }

    .record-bad {
        color: #ff4444;
        font-weight: bold;
    }

    .header-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }

    .stat-box {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }

    h1 {
        color: #667eea;
    }

    h2 {
        color: #764ba2;
        border-bottom: 2px solid #667eea;
        padding-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_league_data():
    """Load league data with caching"""
    try:
        lm = LeagueManager(LEAGUES)
        return lm
    except Exception as e:
        st.error(f"Error loading leagues: {e}")
        return None


def format_record(wins, losses):
    """Format win-loss record with colors"""
    return f"{wins}W - {losses}L"


def get_record_color(wins, losses):
    """Get color based on win percentage"""
    total = wins + losses
    if total == 0:
        return "#999999"
    win_pct = wins / total
    if win_pct >= 0.6:
        return "#00cc44"  # Green for winning record
    elif win_pct >= 0.4:
        return "#ffaa00"  # Orange for .500
    else:
        return "#ff4444"  # Red for losing record


def create_standings_visual(standings_df):
    """Create a visual standings chart"""
    if standings_df.empty:
        return None

    # Create a color column based on standing
    standings_df = standings_df.copy()
    standings_df['color'] = standings_df.apply(
        lambda row: get_record_color(int(row['W-L'].split('-')[0]), int(row['W-L'].split('-')[1])),
        axis=1
    )

    fig = go.Figure(data=[
        go.Bar(
            y=standings_df['Team'],
            x=standings_df['Points For'],
            orientation='h',
            marker=dict(color=standings_df['color']),
            text=standings_df['Points For'],
            textposition='auto',
        )
    ])

    fig.update_layout(
        title="League Points For",
        xaxis_title="Points For",
        yaxis_title="Team",
        height=400,
        showlegend=False,
        template="plotly_white"
    )

    return fig


def main():
    # Header
    st.markdown("""
    <div class="header-banner">
        <h1 style="margin: 0; color: white;">🏈 Fantasy Football Dashboard</h1>
        <p style="margin: 5px 0 0 0;">Your complete fantasy football control center</p>
    </div>
    """, unsafe_allow_html=True)

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
        st.header("📊 League Overview")

        all_teams = league_manager.get_user_teams()

        if not all_teams.empty:
            # Top metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📍 Leagues</h3>
                    <h1>{len(league_manager.leagues)}</h1>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>👥 Teams</h3>
                    <h1>{len(all_teams)}</h1>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                total_wins = all_teams['Wins'].sum()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🏅 Total Wins</h3>
                    <h1>{int(total_wins)}</h1>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                total_pf = all_teams['Points For'].sum()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📈 Total PF</h3>
                    <h1>{total_pf:.0f}</h1>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("Your Teams by League")

            for league in league_manager.leagues.keys():
                league_teams = all_teams[all_teams['League'] == league]

                with st.expander(f"**{league}** ({len(league_teams)} teams)", expanded=True):
                    cols = st.columns(len(league_teams)) if len(league_teams) > 0 else [st.columns(1)]

                    for idx, (_, team) in enumerate(league_teams.iterrows()):
                        with cols[idx]:
                            wins = int(team['Wins'])
                            losses = int(team['Losses'])
                            record_color = get_record_color(wins, losses)

                            st.markdown(f"""
                            <div class="team-card">
                                <h3>{team['Team']}</h3>
                                <p style="font-size: 24px; margin: 10px 0;">
                                    <span style="color: {record_color};">{wins}W - {losses}L</span>
                                </p>
                                <p>📍 Standing: #{team['Standing']}</p>
                                <p>📈 Points: {team['Points For']:.1f}</p>
                            </div>
                            """, unsafe_allow_html=True)
        else:
            st.warning("No team data available")

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
                            use_container_width=True,
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
                use_container_width=True,
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
                    use_container_width=True,
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
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            # Table standings
            st.subheader("Detailed Standings")

            # Add rank column
            standings_display = standings.copy()
            standings_display.insert(0, 'Rank', range(1, len(standings_display) + 1))

            st.dataframe(
                standings_display,
                use_container_width=True,
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
