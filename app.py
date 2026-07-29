import streamlit as st
import pandas as pd
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Cricket Match Win Analyzer",
    page_icon="🏏",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
pipe = pickle.load(open("pipe.pkl", "rb"))

matches = pd.read_csv("data/matches.csv")

teams = sorted([
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
])

cities = sorted(matches['city'].dropna().unique())

# ---------------- CSS ----------------
st.markdown("""
<style>

.stApp{
    background: linear-gradient(135deg,#0f172a,#1e293b,#111827);
    color:white;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:#111827;
}

/* Title */
.main-title{
    font-size:42px;
    font-weight:bold;
    text-align:center;
    color:#FFD700;
}

.subtitle{
    text-align:center;
    color:#d1d5db;
    margin-bottom:30px;
}

/* Cards */
.card{
    background:#1f2937;
    padding:20px;
    border-radius:18px;
    border:1px solid #374151;
    box-shadow:0px 4px 20px rgba(0,0,0,0.3);
}

/* Metric cards */
div[data-testid="metric-container"]{
    background:#1f2937;
    border:1px solid #374151;
    padding:18px;
    border-radius:15px;
}

/* Button */
.stButton>button{
    width:100%;
    background:#f59e0b;
    color:black;
    font-weight:bold;
    border-radius:12px;
    height:55px;
    font-size:18px;
    border:none;
}

.stButton>button:hover{
    background:#facc15;
}

/* Selectbox */
.stSelectbox label{
    color:white;
}

/* Number input */
.stNumberInput label{
    color:white;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown(
    "<h1 class='main-title'>🏏 Cricket Match Win Analyzer</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='subtitle'>AI Powered IPL Winning Probability Predictor</p>",
    unsafe_allow_html=True
)

# ---------------- SIDEBAR ----------------
with st.sidebar:

    st.header("🏏 Match Details")

    batting_team = st.selectbox(
        "Batting Team",
        teams
    )

    bowling_team = st.selectbox(
        "Bowling Team",
        teams
    )

    selected_city = st.selectbox(
        "Venue",
        cities
    )

    target = st.number_input(
        "Target",
        min_value=1
    )

    score = st.number_input(
        "Current Score",
        min_value=0
    )

    overs = st.number_input(
        "Overs Completed",
        min_value=0.0,
        max_value=20.0,
        step=0.1
    )

    wickets = st.number_input(
        "Wickets Fallen",
        min_value=0,
        max_value=10
    )

    predict = st.button("🚀 Predict Winner")

# ---------------- MATCH SUMMARY ----------------
st.markdown("## 📋 Match Summary")

col1, col2 = st.columns(2)

with col1:

    st.markdown(f"""
<div class="card">

### 🏏 Batting Team

**Team :** {batting_team}

**Current Score :** {score}/{wickets}

**Overs :** {overs}

</div>
""", unsafe_allow_html=True)

with col2:

    st.markdown(f"""
<div class="card">

### 🎯 Match Info

**Bowling Team :** {bowling_team}

**Venue :** {selected_city}

**Target :** {target}

</div>
""", unsafe_allow_html=True)

st.divider()
# ---------------- PREDICTION ----------------

if predict:

    runs_left = target - score
    balls_left = 120 - int(overs * 6)
    wickets_left = 10 - wickets

    crr = score / overs if overs > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets_left],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    result = pipe.predict_proba(input_df)
    st.write("Classes:", pipe.classes_)
    st.write("Probabilities:", result)

    loss = result[0][0]
    win = result[0][1]
    st.write(result)
    st.write(input_df)

    # ---------------- MATCH STATS ----------------

    st.subheader("📊 Match Statistics")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Runs Left", runs_left)
    c2.metric("Balls Left", balls_left)
    c3.metric("Current RR", round(crr, 2))
    c4.metric("Required RR", round(rrr, 2))

    st.divider()

    # ---------------- WIN PROBABILITY ----------------

    st.subheader("🏆 Winning Probability")

    st.write(f"### {batting_team}")
    st.progress(float(win))
    st.write(f"**{round(win*100)}%**")

    st.write("")

    st.write(f"### {bowling_team}")
    st.progress(float(loss))
    st.write(f"**{round(loss*100)}%**")

    st.divider()

    # ---------------- WINNER ----------------

    if win > loss:
        st.success(f"🏆 Predicted Winner: {batting_team}")
        st.balloons()
    else:
        st.success(f"🏆 Predicted Winner: {bowling_team}")
        st.balloons()