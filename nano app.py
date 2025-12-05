import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide")
st.title("🏏 Cricket Predictor")

@st.cache_data
def get_data():
    np.random.seed(42)
    data = []
    players = ['Virat Kohli', 'Rohit Sharma', 'KL Rahul']
    for i in range(20):
        for p in players:
            data.append({
                'player': p,
                'match': f'M{i+1}',
                'runs': np.random.choice([10,20,30,40,0], p=[0.2,0.3,0.25,0.15,0.1])
            })
    return pd.DataFrame(data)

df = get_data()

col1, col2 = st.columns(2)
col1.metric("Matches", df['match'].nunique())
col2.metric("Players", len(df['player'].unique()))

st.subheader("🏆 Top Scorers")
stats = df.groupby('player')['runs'].agg(['sum', 'mean']).round(0)
st.dataframe(stats.sort_values('sum', ascending=False))

player = st.selectbox("Select Player", df['player'].unique())
if st.button(f"Predict {player}"):
    player_data = df[df['player'] == player]
    pred = round(player_data['runs'].mean() + np.random.normal(0, 3))
    st.metric("Next Match", pred)
    st.success(f"🎯 {player}: {pred} runs predicted!")

fig = px.bar(stats.sort_values('sum', ascending=False), 
             x='sum', y=stats.index, title="Total Runs")
st.plotly_chart(fig)
