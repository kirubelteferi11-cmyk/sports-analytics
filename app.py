import pandas as pd
import requests
import streamlit as st
import nfl_data_py as nfl
from nba_api.stats.endpoints import leaguegamefinder
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Custom Matchup Analytics", layout="wide")
st.title("Custom Matchup & Performance Dashboard")

# NFL Team Codes Mapping
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
    "Los Angeles Rams": {"code": "LA", "city": "Los Angeles"},
    "Philadelphia Eagles": {"code": "PHI", "city": "Philadelphia"},
    "San Francisco 49ers": {"code": "SF", "city": "Santa Clara"}
}

INTERNATIONAL_VENUES = {
    "Standard Home Venue": None,
    "Melbourne, Australia (MCG)": {"city": "Melbourne", "tz_offset": "+15 hrs"},
    "London, UK (Wembley)": {"city": "London", "tz_offset": "+5 hrs"},
    "Munich, Germany (Allianz Arena)": {"city": "Munich", "tz_offset": "+6 hrs"},
    "São Paulo, Brazil (Arena Corinthians)": {"city": "Sao Paulo", "tz_offset": "+1 hr"}
}

st.sidebar.header("Matchup Configuration")
league = st.sidebar.radio("League", ["NFL", "NBA"])

if league == "NFL":
    col1, col2 = st.sidebar.columns(2)
    home_team_name = col1.selectbox("Home Team", list(NFL_TEAMS.keys()), index=9)
    away_team_name = col2.selectbox("Away Team", list(NFL_TEAMS.keys()), index=11)
    
    venue_selection = st.sidebar.selectbox("Game Venue", list(INTERNATIONAL_VENUES.keys()))
    
    home_code = NFL_TEAMS[home_team_name]["code"]
    away_code = NFL_TEAMS[away_team_name]["code"]
    
    is_international = venue_selection != "Standard Home Venue"
    host_city = INTERNATIONAL_VENUES[venue_selection]["city"] if is_international else NFL_TEAMS[home_team_name]["city"]
    
    if is_international:
        st.header(f"International Series: {away_team_name} vs. {home_team_name}")
        st.info(f"**Neutral Venue:** {venue_selection} | **Time Shift:** {INTERNATIONAL_VENUES[venue_selection]['tz_offset']} relative to US Eastern Time")
    else:
        st.header(f"Matchup: {away_team_name} @ {home_team_name}")

    # 1. Location Weather Tracker
    st.subheader(f"1. Game Location Weather ({host_city})")
    try:
        geo = requests.get(f"https://geocoding-api.open-meteo.com/v1/search?name={host_city}").json()
        if geo.get("results"):
            lat, lon = geo["results"][0]["latitude"], geo["results"][0]["longitude"]
            weather = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true").json()["current_weather"]
            
            w1, w2 = st.columns(2)
            w1.metric("Temperature", f"{weather['temperature']} °C")
            w2.metric("Wind Speed", f"{weather['windspeed']} km/h")
    except Exception as e:
        st.error(f"Could not load weather data: {e}")

    # 2. Active Injuries
    st.subheader("2. Active Injury Reports")
    try:
        injuries = nfl.import_injuries([2024])
        matchup_injuries = injuries[injuries['team'].isin([home_code, away_code])]
        name_col = 'full_name' if 'full_name' in matchup_injuries.columns else 'player_name'
        cols_to_show = [c for c in ['team', name_col, 'position', 'report_primary_injury', 'report_status'] if c in matchup_injuries.columns]
        st.dataframe(matchup_injuries[cols_to_show])
    except Exception as e:
        st.error(f"Could not load injury reports: {e}")

    # 3. Madden Ratings Analysis
    st.subheader("3. Team Madden Ratings Breakdown")
    home_madden_avg, away_madden_avg = 80.0, 80.0  # Fallback averages
    try:
        madden_df = nfl.import_madden_ratings([2024])
        matchup_madden = madden_df[madden_df['team'].isin([home_code, away_code])]
        
        home_madden_avg = matchup_madden[matchup_madden['team'] == home_code]['overall_rating'].mean() or 80.0
        away_madden_avg = matchup_madden[matchup_madden['team'] == away_code]['overall_rating'].mean() or 80.0
        
        m1, m2 = st.columns(2)
        m1.metric(f"{home_team_name} Madden Rating", f"{home_madden_avg:.1f} OVR")
        m2.metric(f"{away_team_name} Madden Rating", f"{away_madden_avg:.1f} OVR")
        
        st.dataframe(matchup_madden[['team', 'full_name', 'position', 'overall_rating']].sort_values(by='overall_rating', ascending=False).head(10))
    except Exception as e:
        st.warning(f"Using standard rating estimates. Details: {e}")

# 3. Team Madden Ratings Analysis
    st.subheader("3. Team Madden Ratings Breakdown")
    
    # Static Madden OVR Ratings Dictionary
    MADDEN_OVR = {
        "ARI": 78, "ATL": 82, "BAL": 89, "BUF": 88, 
        "CAR": 74, "CHI": 80, "DAL": 87, "GB": 84, 
        "KC": 92,  "LA": 85,  "PHI": 89, "SF": 90
    }
    
    home_madden_avg = MADDEN_OVR.get(home_code, 80.0)
    away_madden_avg = MADDEN_OVR.get(away_code, 80.0)
    
    m1, m2 = st.columns(2)
    m1.metric(f"{home_team_name} Madden Rating", f"{home_madden_avg:.1f} OVR")
    m2.metric(f"{away_team_name} Madden Rating", f"{away_madden_avg:.1f} OVR")
