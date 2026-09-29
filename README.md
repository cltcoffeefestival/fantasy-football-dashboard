# 🏈 Fantasy Football Dashboard

A comprehensive web dashboard to analyze your ESPN Fantasy Football teams across multiple leagues. Get trade recommendations, waiver wire pickups, strength of schedule analysis, and more—all in one place.

## Features

- **📊 League Overview**: View all your teams and their standings
- **👥 Team Analysis**: Detailed roster analysis with projected scores
- **📋 Waiver Wire**: Position-specific recommendations for free agent pickups
- **🤝 Trade Finder**: Identify potential trade partners based on team needs
- **🏆 League Standings**: Full league standings and rankings
- **🏥 Injury Report**: Track key player injuries across your leagues
- **📈 Strength of Schedule**: Analyze upcoming matchups

## Prerequisites

- Python 3.8+
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
├── app.py              # Main Streamlit dashboard
├── league_manager.py   # ESPN API integration
├── analyzer.py         # Analysis and recommendations engine
├── config.py          # Configuration (league IDs, settings)
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## Usage

1. **Overview**: See a quick summary of all your teams and leagues
2. **My Teams**: Drill into individual teams and view detailed rosters
3. **Team Analysis**: Get strength of schedule and projected scores
4. **Waiver Wire**: Find best available free agents by position
5. **Trade Finder**: Identify teams with complementary needs
6. **League Standings**: View complete league rankings
7. **Injuries**: Track injured players across the league

## Recommendations Explained

### Waiver Wire Pickups
Shows the highest-projected available player at each position that you might consider adding.

### Trade Partners
Identifies teams whose strengths/weaknesses complement yours, suggesting good trade candidates.

### Strength of Schedule
Shows your team's upcoming matchups to help identify favorable/tough weeks.

## Advanced Features (Coming Soon)

- Historical performance trends
- Player consistency analysis
- Playoff readiness scores
- Trade evaluation (fair value calculator)
- Bench performance metrics
- Bench vs. Start optimization

## Troubleshooting

**"No leagues configured" error:**
- Make sure you've edited `config.py` with valid league IDs

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
