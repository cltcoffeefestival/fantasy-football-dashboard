"""
Configuration for ESPN Fantasy Football Dashboard
Add your league IDs and credentials via environment variables
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Your Fantasy Football Leagues
LEAGUES = [
    {"league_id": 178016325, "year": 2026, "name": "Disgusting Myrtle PT. 2"},
    {"league_id": 1909924039, "year": 2026, "name": "Gambling Chat"},
]

CURRENT_YEAR = 2026

# ESPN Authentication - Load from environment variables
# Set these in Streamlit Cloud secrets or .env file
ESPN_S2 = os.getenv("ESPN_S2", "")
SWID = os.getenv("SWID", "")

# ESPN API settings
ESPN_API_YEAR_RANGE = range(CURRENT_YEAR - 5, CURRENT_YEAR + 1)
