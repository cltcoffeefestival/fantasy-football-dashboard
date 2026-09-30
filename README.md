# 🏈 Fantasy Football Dashboard

A comprehensive web dashboard to analyze your ESPN Fantasy Football teams across multiple leagues. Get trade recommendations, waiver wire pickups, strength of schedule analysis, and more—all in one place.

## Features

- **Dashboard**: team cards for every league, sorted by standing, with your own teams highlighted
- **My Teams**: record, standing, points per week and the roster grouped by position
- **Waiver Wire**: best available free agent at each position, plus a filterable free agent table
- **Trade Finder**: scores trades from the other manager's side with the Manager Acceptance Index (MAI = ΔLineup − Alpha tax − bench clutter − trade structure − asymmetry) and proposes Win-Win, Worth a Shot and Long Shot trades for any team, plus a scorer for trades you type in
- **Team Analysis**: team summary
- **Standings**: points-for chart and full standings for each league

A sidebar picker selects the league and team once for every page. Your team is
detected by matching the `SWID` cookie to team owners. A Refresh button reloads
data from ESPN (league data is otherwise cached for 15 minutes).

## Prerequisites

- Python 3.9+
- pip (Python package manager)
- Your ESPN Fantasy Football league IDs

## Setup Instructions

### 1. Clone/Create Project
```bash
cd fantasy-football-dashboard
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Your Leagues

Edit `config.py` and add your league IDs:

```python
LEAGUES = [
    {"league_id": 12345, "year": 2024, "name": "Main League"},
    {"league_id": 67890, "year": 2024, "name": "Work League"},
]
```

**How to find your league ID:**
1. Go to ESPN.com and navigate to your Fantasy Football league
2. The URL will look like: `https://fantasy.espn.com/football/league?leagueId=12345`
3. The number after `leagueId=` is your league ID

### 5. Run the Dashboard
```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Project Structure

```
fantasy-football-dashboard/
├── app.py               # Streamlit UI
├── assets/style.css     # Shared styling
├── league_manager.py    # ESPN API integration
├── analyzer.py          # Analysis and recommendations
├── trade_finder.py      # Trade engine (MAI scoring, lineups, trade generation)
├── tests/               # pytest suite for the trade engine
├── config.py            # League IDs and settings
├── .streamlit/          # Theme and secrets template
├── .env.example         # Credentials template
└── requirements.txt
```

## Configuration

League IDs live in `config.py`. Credentials come from environment variables:
copy `.env.example` to `.env` locally, or set `ESPN_S2` and `SWID` as secrets
when deploying.

## Deploying to Streamlit Community Cloud

1. Push the repo to GitHub and create an app at [share.streamlit.io](https://share.streamlit.io) with `app.py` as the main file.
2. Under **Advanced settings → Secrets**, add `ESPN_S2` and `SWID` (see `.streamlit/secrets.toml.example`).
3. Restrict viewing to specific emails under the app's sharing settings, since it uses your ESPN login.

If leagues stop loading, your ESPN cookies have probably expired: refresh them.

## Troubleshooting

**"No leagues configured" error:**
- Make sure `config.py` has valid league IDs; failed leagues are listed in the sidebar

**"Failed to load league" error:**
- Verify your league ID is correct
- Ensure the league is public or you have access to it
- Check your internet connection

**No roster/player data:**
- Some ESPN leagues may restrict API access
- Try accessing the league through ESPN's website to verify access

## Dependencies

- **streamlit**: Web app framework
- **espn-api**: ESPN Fantasy API client
- **pandas**: Data manipulation
- **plotly**: Data visualization

## License

Open source - feel free to modify and use as you wish!

## Contributing

Found a bug or have a feature idea? Feel free to extend this project!

### Ideas for Enhancement:
- Add historical trends and charts
- Implement trade value calculator
- Create notifications for high-value free agents
- Add season projections
- Build player comparison tools
- Add custom scoring analysis
