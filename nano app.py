import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Load model (update path to your saved model)
@st.cache_resource
def load_model():
    return joblib.load('cricket_rf_model.joblib')

model = load_model()

# Player mapping (from your notebook)
PLAYERS = {
    0: 'Virat Kohli', 1: 'Rohit Sharma', 2: 'Suryakumar Yadav', 3: 'KL Rahul',
    4: 'Jos Buttler', 5: 'Babar Azam', 6: 'Glenn Maxwell', 7: 'Hardik Pandya'
}

st.set_page_config(page_title="Cricket Form Predictor", layout="wide")

st.title("🏏 Cricket Player Form Predictor")
st.markdown("**Predict next innings runs using Random Forest (MAE: 11.4 runs)**")

# Sidebar inputs
st.sidebar.header("Player Input")
player_name = st.sidebar.selectbox("Select Player", list(PLAYERS.values()))
recent_innings = st.sidebar.number_input(
    "Last 5 Innings (comma separated)", 
    value="30,25,35,22,28",
    help="Enter last 5 innings runs (use averages if missing)"
)
career_avg = st.sidebar.number_input("Career Average", value=28.0, step=0.5)

if st.sidebar.button("Predict Next Innings"):
    # Parse recent innings
    try:
        recent = [float(x.strip()) for x in recent_innings.split(',')]
        player_id = list(PLAYERS.values()).index(player_name)
        
        # Feature engineering (matches notebook)
        prev_runs = recent[-1]
        roll_avg3 = np.mean(recent[-3:])
        roll_avg5 = np.mean(recent[-5:])
        
        features = np.array([[player_id, prev_runs, roll_avg3, roll_avg5, career_avg]])
        prediction = model.predict(features)[0]
        
        st.sidebar.success(f"**Predicted: {prediction:.1f} runs**")
        st.session_state.prediction = prediction
        st.session_state.features = features[0]
        st.session_state.recent = recent
        
    except:
        st.sidebar.error("Enter valid comma-separated numbers")

# Main dashboard
col1, col2 = st.columns(2)

with col1:
    st.metric("Predicted Runs", f"{st.session_state.get('prediction', 28):.1f}", delta="±11.4")
    
    if 'features' in st.session_state:
        st.subheader("Feature Breakdown")
        feat_df = pd.DataFrame({
            'Feature': ['Player ID', 'Prev Runs', '3GA', '5GA', 'Career Avg'],
            'Value': st.session_state.features
        })
        st.dataframe(feat_df)

with col2:
    if 'recent' in st.session_state:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, len(st.session_state.recent)+2)),
            y=st.session_state.recent + [st.session_state.prediction],
            mode='lines+markers+text',
            name='Runs',
            text=[f"{r:.0f}" for r in st.session_state.recent + [st.session_state.prediction]],
            textposition="top center"
        ))
        fig.update_layout(title="Recent + Predicted", xaxis_title="Innings", yaxis_title="Runs")
        st.plotly_chart(fig, use_container_width=True)

# Model performance
st.subheader("Model Performance")
metrics = pd.DataFrame({
    'Metric': ['MAE', 'RMSE', 'R²'],
    'Test Set': ['11.44', '13.90', '-0.013']
})
st.dataframe(metrics)

st.markdown("---")
st.caption("Built with your Random Forest model from Analysis_FINAL.ipynb")
