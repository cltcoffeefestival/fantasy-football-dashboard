"""
League Manager - Handles fetching league and team data from ESPN API
"""
from espn_api.football import League
from config import LEAGUES, ESPN_S2, SWID
import pandas as pd
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LeagueManager:
    """Manages ESPN fantasy football leagues and team data"""

    def __init__(self, leagues_config: List[Dict[str, Any]] = None):
        self.leagues_config = leagues_config or LEAGUES
        self.leagues = {}
        self.teams_data = {}
        self.load_leagues()

    def load_leagues(self):
        """Load all configured leagues from ESPN API"""
        for league_config in self.leagues_config:
            league_id = league_config["league_id"]
            year = league_config.get("year", 2024)
            name = league_config.get("name", f"League {league_id}")

            try:
                league = League(league_id=league_id, year=year, espn_s2=ESPN_S2, swid=SWID)
                key = f"{name} ({year})"
                self.leagues[key] = league
                logger.info(f"Loaded league: {key}")
            except Exception as e:
                logger.error(f"Failed to load league {league_id}: {e}")

    def get_user_teams(self) -> pd.DataFrame:
        """Get all user teams across all leagues"""
        teams = []

        for league_name, league in self.leagues.items():
            for team in league.teams:
                teams.append({
                    "League": league_name,
                    "Team": team.team_name,
                    "Owner": "You",
                    "Wins": team.wins,
                    "Losses": team.losses,
                    "Points For": round(team.points_for, 1),
                    "Points Against": round(team.points_against, 1),
                    "Standing": team.standing,
                    "Team ID": team.team_id,
                    "League ID": league.league_id,
                })

        return pd.DataFrame(teams)

    def get_team_roster(self, league_name: str, team_id: int) -> pd.DataFrame:
        """Get roster for a specific team"""
        league = self.leagues.get(league_name)
        if not league:
            return pd.DataFrame()

        team = next((t for t in league.teams if t.team_id == team_id), None)
        if not team:
            return pd.DataFrame()

        roster = []
        for player in team.roster:
            try:
                pro_team = getattr(player, 'proTeam', None) or getattr(player, 'nfl_team', 'N/A')
                avg_pts = getattr(player, 'avg_points', 0) or 0
                proj = getattr(player, 'projected', 0) or 0

                roster.append({
                    "Name": player.name,
                    "Position": player.position,
                    "Team": pro_team if pro_team else 'N/A',
                    "Avg Points": round(float(avg_pts), 1) if avg_pts else 0,
                    "Projected": round(float(proj), 1) if proj else 0,
                })
            except Exception as e:
                logger.warning(f"Error processing player {player.name}: {e}")
                continue

        return pd.DataFrame(roster)

    def get_free_agents(self, league_name: str, limit: int = 50) -> pd.DataFrame:
        """Get free agents/waiver wire for a league"""
        league = self.leagues.get(league_name)
        if not league:
            return pd.DataFrame()

        free_agents = []
        try:
            for player in league.free_agents(size=limit):
                try:
                    pro_team = getattr(player, 'proTeam', None) or getattr(player, 'nfl_team', 'N/A')
                    avg_pts = getattr(player, 'avg_points', 0) or 0
                    proj = getattr(player, 'projected', 0) or 0

                    free_agents.append({
                        "Name": player.name,
                        "Position": player.position,
                        "Team": pro_team if pro_team else 'N/A',
                        "Avg Points": round(float(avg_pts), 1) if avg_pts else 0,
                        "Projected": round(float(proj), 1) if proj else 0,
                    })
                except Exception as e:
                    logger.warning(f"Error processing free agent {player.name}: {e}")
                    continue
        except Exception as e:
            logger.error(f"Error getting free agents: {e}")
            return pd.DataFrame()

        return pd.DataFrame(free_agents)

    def get_league_standings(self, league_name: str) -> pd.DataFrame:
        """Get league standings"""
        league = self.leagues.get(league_name)
        if not league:
            return pd.DataFrame()

        standings = []
        for team in sorted(league.teams, key=lambda x: x.standing):
            standings.append({
                "Team": team.team_name,
                "W-L": f"{team.wins}-{team.losses}",
                "Points For": round(team.points_for, 1),
                "Points Against": round(team.points_against, 1),
                "Standing": team.standing,
            })

        return pd.DataFrame(standings)

    def get_matchups(self, league_name: str) -> pd.DataFrame:
        """Get current/upcoming matchups for a league"""
        league = self.leagues.get(league_name)
        if not league:
            return pd.DataFrame()

        matchups = []
        try:
            for matchup in league.matchups():
                matchups.append({
                    "Week": matchup.week,
                    "Home": matchup.home_team.team_name,
                    "Away": matchup.away_team.team_name,
                    "Home Score": matchup.home_score or 0,
                    "Away Score": matchup.away_score or 0,
                })
        except Exception as e:
            logger.error(f"Error getting matchups: {e}")

        return pd.DataFrame(matchups)
