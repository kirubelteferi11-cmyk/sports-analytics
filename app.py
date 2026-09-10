import pandas as pd
import requests
import streamlit as st
import nfl_data_py as nfl
from nba_api.stats.endpoints import leaguegamefinder
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Sports Analytics Machine Learning Engine", layout="wide")
st.title("Comprehensive Sports Analytics & Prediction Engine")

# Complete 32 NFL Team Mapping
NFL_TEAMS = {
    "Arizona Cardinals": {"code": "ARI", "city": "Glendale"},
    "Atlanta Falcons": {"code": "ATL", "city": "Atlanta"},
    "Baltimore Ravens": {"code": "BAL", "city": "Baltimore"},
    "Buffalo Bills": {"code": "BUF", "city": "Orchard Park"},
    "Carolina Panthers": {"code": "CAR", "city": "Charlotte"},
    "Chicago Bears": {"code": "CHI", "city": "Chicago"},
    "Cincinnati Bengals": {"code": "CIN", "city": "Cincinnati"},
    "Cleveland Browns": {"code": "CLE", "city": "Cleveland"},
    "Dallas Cowboys": {"code": "DAL", "city": "Arlington"},
    "Denver Broncos": {"code": "DEN", "city": "Denver"},
    "Detroit Lions": {"code": "DET", "city": "Detroit"},
    "Green Bay Packers": {"code": "GB", "city": "Green Bay"},
    "Houston Texans": {"code": "HOU", "city": "Houston"},
    "Indianapolis Colts": {"code": "IND", "city": "Indianapolis"},
    "Jacksonville Jaguars": {"code": "JAX", "city": "Jacksonville"},
    "Kansas City Chiefs": {"code": "KC", "city": "Kansas City"},
    "Las Vegas Raiders": {"code": "LV", "city": "Las Vegas"},
    "Los Angeles Chargers": {"code": "LAC", "city": "Inglewood"},
    "Los Angeles Rams": {"code": "LA", "city": "Inglewood"},
    "Miami Dolphins": {"code": "MIA", "city": "Miami Gardens"},
    "Minnesota Vikings": {"code": "MIN", "city": "Minneapolis"},
    "New England Patriots": {"code": "NE", "city": "Foxborough"},
    "New Orleans Saints": {"code": "NO", "city": "New Orleans"},
    "New York Giants": {"code": "NYG", "city": "East Rutherford"},
    "New York Jets": {"code": "NYJ", "city": "East Rutherford"},
    "Philadelphia Eagles": {"code": "PHI", "city": "Philadelphia"},
    "Pittsburgh Steelers": {"code": "PIT", "city": "Pittsburgh"},
    "San Francisco 49ers": {"code": "SF", "city": "Santa Clara"},
    "Seattle Seahawks": {"code": "SEA", "city": "Seattle"},
    "Tampa Bay Buccaneers": {"code": "TB", "city": "Tampa"},
    "Tennessee Titans": {"code": "TEN", "city": "Nashville"},
    "Washington Commanders": {"code": "WAS", "city": "Landover"}
}

# Baseline Madden Overall Ratings Dataset (32 Teams)
MADDEN_OVR = {
    "ARI": 78, "ATL": 82, "BAL": 89, "BUF": 88, "CAR": 74, "CHI": 80,
    "CIN": 85, "CLE": 83, "DAL": 87, "DEN": 77, "DET": 86, "GB": 84,
    "HOU": 83, "IND": 81, "JAX": 80, "KC": 92,  "LV": 81,  "LAC": 82,
    "LA": 85,  "MIA": 86, "MIN": 81, "NE": 76,  "NO": 79,  "NYG": 78,
    "NYJ": 84, "PHI": 89, "PIT": 82, "SF": 90,  "SEA": 81, "TB": 81,
    "TEN": 77, "WAS": 79
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
    home_team_name = col1.selectbox("Home Team", list(NFL_TEAMS.keys()), index=15) # Chiefs
    away_team_name = col2.selectbox("Away Team", list(NFL_TEAMS.keys()), index=27) # 49ers
    venue_selection = st.sidebar.selectbox("Game Venue", list(INTERNATIONAL_VENUES.keys()))
    
    home_code = NFL_TEAMS[home_team_name]["code"]
    away_code = NFL_TEAMS[away_team_name]["code"]
    
    is_international = venue_selection != "Standard Home Venue"
    host_city = INTERNATIONAL_VENUES[venue_selection]["city"] if is_international else NFL_TEAMS[home_team_name]["city"]
    
    if is_international:
        st.header(f"International Matchup: {away_team_name} vs. {home_team_name}")
        st.info(f"**Neutral Venue:** {venue_selection} | **Time Offset:** {INTERNATIONAL_VENUES[venue_selection]['tz_offset']} relative to US Eastern Time")
    else:
        st.header(f"Matchup: {away_team_name} @ {home_team_name}")

    # Section 1: Weather Factor
    st.subheader(f"1. Stadium Location & Weather Impact ({host_city})")
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

    # Section 2: Active Injury Reports
    st.subheader("2. Active Injury Status")
    try:
        injuries = nfl.import_injuries([2024])
        matchup_injuries = injuries[injuries['team'].isin([home_code, away_code])]
        name_col = 'full_name' if 'full_name' in matchup_injuries.columns else 'player_name'
        cols = [c for c in ['team', name_col, 'position', 'report_primary_injury', 'report_status'] if c in matchup_injuries.columns]
        st.dataframe(matchup_injuries[cols])
    except Exception as e:
        st.error(f"Could not load injury reports: {e}")

    # Section 3: Head-to-Head Meetings (2022–2026)
    st.subheader("3. Historical Matchups (2022–2026)")
    try:
        schedules = nfl.import_schedules([2022, 2023, 2024, 2025, 2026])
        h2h = schedules[
            ((schedules['home_team'] == home_code) & (schedules['away_team'] == away_code)) |
            ((schedules['home_team'] == away_code) & (schedules['away_team'] == home_code))
        ]
        st.dataframe(h2h[['season', 'week', 'home_team', 'away_team', 'home_score', 'away_score', 'roof']])
    except Exception as e:
        st.error(f"Could not load schedule data: {e}")

    # Section 4: Team Talent Breakdown (Madden Ratings)
    st.subheader("4. Team Madden Talent Breakdown")
    home_madden = MADDEN_OVR.get(home_code, 80)
    away_madden = MADDEN_OVR.get(away_code, 80)
    
    m1, m2 = st.columns(2)
    m1.metric(f"{home_team_name} Madden Rating", f"{home_madden:.1f} OVR")
    m2.metric(f"{away_team_name} Madden Rating", f"{away_madden:.1f} OVR")

    # Section 5: Machine Learning Conclusion & Win Probability
    st.subheader("5. AI Model Conclusion & Projection")
    try:
        train_schedules = nfl.import_schedules([2022, 2023, 2024, 2025, 2026]).dropna(subset=['home_score', 'away_score'])
        train_schedules['home_win'] = (train_schedules['home_score'] > train_schedules['away_score']).astype(int)
        
        X = train_schedules[['home_score', 'away_score']]
        y = train_schedules['home_win']
        
        model = LogisticRegression()
        model.fit(X, y)
        
        home_avg = train_schedules[train_schedules['home_team'] == home_code]['home_score'].mean() or 20.0
        away_avg = train_schedules[train_schedules['away_team'] == away_code]['away_score'].mean() or 20.0
        
        base_win_prob = 0.50 + ((home_avg - away_avg) * 0.015) if is_international else model.predict_proba([[home_avg, away_avg]])[0][1]
        
        madden_diff = home_madden - away_madden
        final_win_prob = base_win_prob + (madden_diff * 0.02)
        final_win_prob = max(0.05, min(0.95, final_win_prob))
        
        predicted_winner = home_team_name if final_win_prob > 0.5 else away_team_name
        confidence = final_win_prob * 100 if final_win_prob > 0.5 else (1 - final_win_prob) * 100
        
        st.success(f"**Projected Winner:** {predicted_winner} ({confidence:.1f}% confidence)")
        
        st.markdown(f"""
        **AI Conclusion Factors:**
        * **Madden Talent Impact:** {home_team_name if madden_diff > 0 else away_team_name} holds a **+{abs(madden_diff):.1f} OVR** talent advantage.
        * **Historical Metrics:** Scoring metrics derive from **2022–2026** match results ({home_avg:.1f} vs {away_avg:.1f} PPG).
        * **Venue Influence:** Location set to **{host_city}** ({'Neutral International Venue' if is_international else 'Standard Home Stadium Advantage'}).
        """)

    except Exception as e:
        st.error(f"Could not calculate AI prediction: {e}")

elif league == "NBA":
    st.header("NBA Matchup Intelligence")
    try:
        games = leaguegamefinder.LeagueGameFinder(league_id_nullable='00').get_data_frames()[0]
        st.dataframe(games[['GAME_DATE', 'TEAM_NAME', 'MATCHUP', 'WL', 'PTS']].head(20))
    except Exception as e:
        st.error(f"Could not load NBA data: {e}")
