import pandas as pd
import requests
import streamlit as st

# Data sources for NBA and NFL
from nba_api.stats.endpoints import leaguegamefinder
import nfl_data_py as nfl

st.title("NBA & NFL Performance Analytics Engine")

league = st.sidebar.selectbox("Select League", ["NFL", "NBA"])

if league == "NFL":
    st.header("NFL Performance & Environment Tracker")
    
    # 1. Fetch NFL Data (Injuries & Schedules)
    st.subheader("1. Recent Injury Reports")
    try:
        # Pull recent injuries
        injuries = nfl.import_injuries([2024])
        st.dataframe(injuries[['player_name', 'team', 'position', 'report_primary_injury', 'report_status']].head(10))
    except Exception as e:
        st.write("Error loading injury data:", e)

    st.subheader("2. Past Game Performance")
    try:
        schedules = nfl.import_schedules([2024])
        st.dataframe(schedules[['home_team', 'away_team', 'home_score', 'away_score', 'roof', 'surface']].head(10))
    except Exception as e:
        st.write("Error loading schedule data:", e)

    # 3. Weather Integration Example
    st.subheader("3. Game Location Weather Tracker")
    city = st.text_input("Enter Game City (e.g., Green Bay):", "Green Bay")
    if st.button("Check Current Weather"):
        # Fetching current weather via Open-Meteo (Free Open API)
        geo_res = requests.get(f"https://geocoding-api.open-meteo.com/v1/search?name={city}").json()
        if geo_res.get("results"):
            lat = geo_res["results"][0]["latitude"]
            lon = geo_res["results"][0]["longitude"]
            weather_res = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true").json()
            cw = weather_res["current_weather"]
            
            st.metric(label="Temperature", value=f"{cw['temperature']} °C")
            st.metric(label="Wind Speed", value=f"{cw['windspeed']} km/h")
        else:
            st.write("City not found.")

elif league == "NBA":
    st.header("NBA Matchup & Performance Tracker")
    
    st.subheader("Recent Team Games")
    try:
        # Fetch game logs for NBA teams
        game_finder = leaguegamefinder.LeagueGameFinder(league_id_nullable='00')
        games = game_finder.get_data_frames()[0]
        st.dataframe(games[['GAME_DATE', 'TEAM_NAME', 'MATCHUP', 'WL', 'PTS']].head(15))
    except Exception as e:
        st.write("Error loading NBA data:", e)
