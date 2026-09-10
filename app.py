import pandas as pd
import requests
import streamlit as st
import nfl_data_py as nfl
from nba_api.stats.endpoints import leaguegamefinder

st.set_page_config(page_title="Custom Matchup Analytics", layout="wide")
st.title("Custom Matchup & Performance Dashboard")

# NFL Team Codes Mapping for Data Filtering
NFL_TEAMS = {
    "Arizona Cardinals": {"code": "ARI", "city": "Glendale"},
    "Atlanta Falcons": {"code": "ATL", "city": "Atlanta"},
    "Baltimore Ravens": {"code": "BAL", "city": "Baltimore"},
    "Buffalo Bills": {"code": "BUF", "city": "Orchard Park"},
    "Carolina Panthers": {"code": "CAR", "city": "Charlotte"},
    "Chicago Bears": {"code": "CHI", "city": "Chicago"},
    "Dallas Cowboys": {"code": "DAL", "city": "Arlington"},
    "Green Bay Packers": {"code": "GB", "city": "Green Bay"},
    "Kansas City Chiefs": {"code": "KC", "city": "Kansas City"},
    "Philadelphia Eagles": {"code": "PHI", "city": "Philadelphia"},
    "San Francisco 49ers": {"code": "SF", "city": "Santa Clara"}
}

st.sidebar.header("Matchup Configuration")
league = st.sidebar.radio("League", ["NFL", "NBA"])

if league == "NFL":
    col1, col2 = st.sidebar.columns(2)
    home_team_name = col1.selectbox("Home Team", list(NFL_TEAMS.keys()), index=7)
    away_team_name = col2.selectbox("Away Team", list(NFL_TEAMS.keys()), index=8)
    
    home_code = NFL_TEAMS[home_team_name]["code"]
    away_code = NFL_TEAMS[away_team_name]["code"]
    st.header(f"Matchup: {away_team_name} @ {home_team_name}")

    # 1. Weather at Home Team Location
    st.subheader("1. Stadium Location & Live Weather")
    stadium_city = NFL_TEAMS[home_team_name]["city"]
    try:
        geo = requests.get(f"https://geocoding-api.open-meteo.com/v1/search?name={stadium_city}").json()
        if geo.get("results"):
            lat, lon = geo["results"][0]["latitude"], geo["results"][0]["longitude"]
            weather = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true").json()["current_weather"]
            
            w1, w2 = st.columns(2)
            w1.metric("Temperature", f"{weather['temperature']} °C")
            w2.metric("Wind Speed", f"{weather['windspeed']} km/h")
    except Exception as e:
        st.error(f"Could not load weather data: {e}")

    # 2. Team Injury Reports
    st.subheader("2. Active Injury Reports")
    try:
        injuries = nfl.import_injuries([2024])
        matchup_injuries = injuries[injuries['team'].isin([home_code, away_code])]
        st.dataframe(matchup_injuries[['team', 'player_name', 'position', 'report_primary_injury', 'report_status']])
    except Exception as e:
        st.error(f"Could not load injury reports: {e}")

    # 3. Past Head-to-Head Matchups
    st.subheader("3. Past Head-to-Head Meetings")
    try:
        schedules = nfl.import_schedules([2022, 2023, 2024])
        h2h = schedules[
            ((schedules['home_team'] == home_code) & (schedules['away_team'] == away_code)) |
            ((schedules['home_team'] == away_code) & (schedules['away_team'] == home_code))
        ]
        st.dataframe(h2h[['season', 'week', 'home_team', 'away_team', 'home_score', 'away_score', 'roof']])
    except Exception as e:
        st.error(f"Could not load past matchups: {e}")

    # 4. Recent Weekly Player Performance
    st.subheader("4. Recent Player Performance Data")
    try:
        weekly_stats = nfl.import_weekly_data([2024])
        team_players = weekly_stats[weekly_stats['recent_team'].isin([home_code, away_code])]
        st.dataframe(team_players[['recent_team', 'player_name', 'position', 'passing_yards', 'rushing_yards', 'receiving_yards', 'fantasy_points']])
    except Exception as e:
        st.error(f"Could not load player stats: {e}")

elif league == "NBA":
    st.header("NBA Matchup Intelligence")
    try:
        games = leaguegamefinder.LeagueGameFinder(league_id_nullable='00').get_data_frames()[0]
        st.dataframe(games[['GAME_DATE', 'TEAM_NAME', 'MATCHUP', 'WL', 'PTS']].head(20))
    except Exception as e:
        st.error(f"Could not load NBA data: {e}")
