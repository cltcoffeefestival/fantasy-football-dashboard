"""
Analyzer - Team analysis and recommendations (trades, pickups, etc.)
"""
import pandas as pd
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class TeamAnalyzer:
    """Analyzes fantasy teams and provides recommendations"""

    def __init__(self, league_manager):
        self.league_manager = league_manager

    def recommend_waiver_pickups(
        self, league_name: str, team_id: int, pool_size: int = 100
    ) -> pd.DataFrame:
        """Recommend high-value waiver wire pickups based on avg points"""
        try:
            free_agents = self.league_manager.get_free_agents(league_name, limit=pool_size)

            if free_agents.empty:
                return pd.DataFrame()

            # Get best available by position
            recommendations = []
            for position in ["QB", "RB", "WR", "TE", "DEF", "K"]:
                position_agents = free_agents[free_agents["Position"] == position]
                if not position_agents.empty:
                    # Sort by avg points first, then projected
                    top_agent = position_agents.nlargest(1, "Avg Points")
                    if not top_agent.empty:
                        agent = top_agent.iloc[0]
                        recommendations.append({
                            "Position": position,
                            "Player": agent["Name"],
                            "Team": agent["Team"],
                            "Avg Points": agent["Avg Points"],
                            "Projected": agent["Projected"],
                        })

            return pd.DataFrame(recommendations)
        except Exception as e:
            logger.error(f"Error getting waiver recommendations: {e}")
            return pd.DataFrame()

    def analyze_strength_of_schedule(self, league_name: str, weeks_ahead: int = 4) -> pd.DataFrame:
        """Analyze upcoming strength of schedule for all teams"""
        try:
            matchups = self.league_manager.get_matchups(league_name)

            if matchups.empty:
                return pd.DataFrame()

            # Show next N weeks of matchups
            return matchups.head(weeks_ahead * 6)  # ~6 matchups per week
        except Exception as e:
            logger.error(f"Error analyzing strength of schedule: {e}")
            return pd.DataFrame()

    def find_trade_opportunities(
        self, league_name: str, team_id: int
    ) -> Dict[str, List[Dict]]:
        """Find potential trade partners based on roster strength"""
        try:
            league = self.league_manager.leagues.get(league_name)
            if not league:
                return {"trade_partners": []}

            my_team = next((t for t in league.teams if t.team_id == team_id), None)
            my_roster = self.league_manager.get_team_roster(league_name, team_id)

            if my_team is None or my_roster.empty:
                return {"trade_partners": []}

            # Simple trade opportunity finder: teams close in standings
            trade_partners = []
            for opponent in league.teams:
                if opponent.team_id == team_id:
                    continue

                # Find teams with complementary records
                trade_match = {
                    "Team": opponent.team_name,
                    "Record": f"{opponent.wins}-{opponent.losses}",
                    "Points For": round(opponent.points_for, 1),
                    "Standing": opponent.standing,
                }
                trade_partners.append(trade_match)

            # Sort by closeness in record
            trade_partners.sort(
                key=lambda x: abs((my_team.wins - int(x["Record"].split("-")[0])))
            )

            return {"trade_partners": trade_partners[:5]}  # Top 5 closest teams

        except Exception as e:
            logger.error(f"Error finding trade opportunities: {e}")
            return {"trade_partners": []}

    def get_injury_report(self, league_name: str) -> pd.DataFrame:
        """Get injury report - simplified version"""
        try:
            league = self.league_manager.leagues.get(league_name)
            if not league:
                return pd.DataFrame()

            # Get all rostered players and check for injury attributes
            injuries = []
            for team in league.teams:
                for player in team.roster:
                    injury_status = getattr(player, 'injuryStatus', None)
                    if injury_status and injury_status != 'ACTIVE':
                        pro_team = getattr(player, 'proTeam', None) or getattr(player, 'nfl_team', 'N/A')
                        injuries.append({
                            "Player": player.name,
                            "Position": player.position,
                            "Team": pro_team if pro_team else 'N/A',
                            "Status": injury_status,
                        })

            return pd.DataFrame(injuries) if injuries else pd.DataFrame()
        except Exception as e:
            logger.error(f"Error getting injury report: {e}")
            return pd.DataFrame()

    def get_team_summary(self, league_name: str, team_id: int) -> Dict:
        """Get comprehensive summary for a team"""
        try:
            league = self.league_manager.leagues.get(league_name)
            team = next((t for t in league.teams if t.team_id == team_id), None)

            if not team:
                return {}

            roster = self.league_manager.get_team_roster(league_name, team_id)
            total_games = team.wins + team.losses
            ppw = (team.points_for / total_games) if total_games > 0 else 0

            summary = {
                "Team": team.team_name,
                "Record": f"{team.wins}-{team.losses}",
                "Standing": team.standing,
                "Points For": round(team.points_for, 1),
                "Points Against": round(team.points_against, 1),
                "PPW": round(ppw, 1),
                "Roster Count": len(roster) if not roster.empty else 0,
            }

            return summary
        except Exception as e:
            logger.error(f"Error getting team summary: {e}")
            return {}

    def get_league_insights(self, league_name: str) -> Dict:
        """Get basic league statistics"""
        try:
            league = self.league_manager.leagues.get(league_name)
            if not league:
                return {}

            standings = self.league_manager.get_league_standings(league_name)

            if standings.empty:
                return {}

            insights = {
                "Total Teams": len(league.teams),
                "League Size": len(league.teams),
                "Current Week": getattr(league, 'current_week', 'N/A'),
            }

            return insights
        except Exception as e:
            logger.error(f"Error getting league insights: {e}")
            return {}
