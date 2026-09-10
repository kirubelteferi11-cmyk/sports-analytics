import pandas as pd
import requests
import streamlit as st
import nfl_data_py as nfl
from nba_api.stats.endpoints import leaguegamefinder
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Custom Matchup Analytics", layout="wide")
st.title("Custom Matchup & Performance Dashboard")

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

    # 1. Weather
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

    # 2. Injuries
    st.subheader("2. Active Injury Reports")
    try:
        injuries = nfl.import_injuries([2024])
        matchup_injuries = injuries[injuries['team'].isin([home_code, away_code])]
        st.dataframe(matchup_injuries[['team', 'player_name', 'position', 'report_primary_injury', 'report_status']])
    except Exception as e:
        st.error(f"Could not load injury reports: {e}")

    # 3. Head-to-Head
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

    # 4. Player Stats
    st.subheader("4. Recent Player Performance Data")
    try:
        weekly_stats = nfl.import_weekly_data([2024])
        team_players = weekly_stats[weekly_stats['recent_team'].isin([home_code, away_code])]
        st.dataframe(team_players[['recent_team', 'player_name', 'position', 'passing_yards', 'rushing_yards', 'receiving_yards', 'fantasy_points']])
    except Exception as e:
        st.error(f"Could not load player stats: {e}")

    # 5. AI Winner Prediction
    st.subheader("5. AI Projected Winner")
    try:
        train_schedules = nfl.import_schedules([2022, 2023, 2024]).dropna(subset=['home_score', 'away_score'])
        train_schedules['home_win'] = (train_schedules['home_score'] > train_schedules['away_score']).astype(int)
        
        X = train_schedules[['home_score', 'away_score']]
        y = train_schedules['home_win']
        
        model = LogisticRegression()
        model.fit(X, y)
        
        home_avg = train_schedules[train_schedules['home_team'] == home_code]['home_score'].mean() or 20.0
        away_avg = train_schedules[train_schedules['away_team'] == away_code]['away_score'].mean() or 20.0
        
        win_prob = model.predict_proba([[home_avg, away_avg]])[0][1]
        
        if win_prob > 0.5:
            st.success(f"**Predicted Winner:** {home_team_name} ({win_prob*100:.1f}% confidence)")
        else:
            st.success(f"**Predicted Winner:** {away_team_name} ({(1-win_prob)*100:.1f}% confidence)")

    except Exception as e:
        st.error(f"Could not calculate AI prediction: {e}")

elif league == "NBA":
    st.header("NBA Matchup Intelligence")
    try:
        games = leaguegamefinder.LeagueGameFinder(league_id_nullable='00').get_data_frames()[0]
        st.dataframe(games[['GAME_DATE', 'TEAM_NAME', 'MATCHUP', 'WL', 'PTS']].head(20))
    except Exception as e:
        st.error(f"Could not load NBA data: {e}")
