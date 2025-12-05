import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import zipfile
import json
import tempfile
import os
from pathlib import Path
import warnings

st.set_page_config(page_title="🏏 Cricket Predictor", layout="wide")
warnings.filterwarnings('ignore')

st.title("🏏 YOUR Cricket Form Predictor")

@st.cache_data
def create_sample_data():
    np.random.seed(42)
    players = ['Virat Kohli', 'Rohit Sharma', 'KL Rahul', 'Suryakumar Yadav']
    data = []
    for i in range(30):
        for player in players:
            data.append({
                'player': player,
                'match_id': f'M{i+1}',
                'runs': np.random.choice([0,5,10,15,25,35,45], p=[0.3,0.2,0.15,0.15,0.1,0.05,0.05]),
                'is_four': np.random.randint(0,5),
                'is_six': np.random.randint(0,3)
            })
    return pd.DataFrame(data)

def rf_prepare_features(df):
    features, targets = [], []
    for player in df['player'].unique():
        player_df = df[df['player'] == player].sort_values('match_id')
        for i in range(5, len(player_df)):
            prev = player_df.iloc[:i]
            curr = player_df.iloc[i]
            career_avg = prev['runs'].mean()
            recent_avg = prev.tail(5)['runs'].mean()
            feature = [career_avg, recent_avg, len(prev), 
                      prev.tail(5)['is_four'].sum(), prev.tail(5)['is_six'].sum()]
            features.append(feature)
            targets.append(curr['runs'])
    return np.array(features), np.array(targets)

# Load data
if 'df' not in st.session_state:
    df = create_sample_data()
    st.session_state.df = df

df = st.session_state.df

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Deliveries", len(df))
col2.metric("Players", df['player'].nunique())
col3.metric("Matches", df['match_id'].nunique())

# Top Players
st.subheader("🏆 Top Players")
player_stats = df.groupby('player').agg({
    'runs': ['sum', 'mean'],
    'is_four': 'sum',
    'is_six': 'sum'
}).round(1)
player_stats.columns = ['Total', 'Avg', '4s', '6s']
st.dataframe(player_stats.sort_values('Total', ascending=False))

# Train Model
if st.button("🚀 Train YOUR Model", type="primary"):
    with st.spinner("Training Random Forest..."):
        X, y = rf_prepare_features(df)
        model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=1)
        model.fit(X, y)
        st.session_state.model = model
        st.success("✅ Model trained!")

# Player Prediction
st.markdown("## 🎯 Predict Next Match")
if 'model' in st.session_state:
    selected_player = st.selectbox("Select Player", df['player'].unique())
    
    if st.button(f"🔮 Predict **{selected_player}**", type="primary"):
        player_df = df[df['player'] == selected_player]
        career_avg = player_df['runs'].mean()
        recent_avg = player_df.tail(5)['runs'].mean()
        
        pred_features = np.array([[career_avg, recent_avg, len(player_df), 
                                 player_df.tail(5)['is_four'].sum(), 
                                 player_df.tail(5)['is_six'].sum()]])
        
        prediction = st.session_state.model.predict(pred_features)[0]
        st.metric("Predicted Runs", f"{round(prediction)}")
        st.success(f"🎯 **{selected_player}**: {round(prediction)} runs predicted!")
        
        # Player history
        fig = px.line(player_df.sort_values('match_id').tail(10), 
                     x='match_id', y='runs', title=f"{selected_player} Recent Form")
        st.plotly_chart(fig, height=400)
else:
    st.info("👆 Click 'Train YOUR Model' first!")

st.markdown("---")
st.caption("✅ Uses YOUR exact feature engineering + Random Forest model")
