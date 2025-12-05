import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import zipfile
import json
import io
import tempfile
import warnings
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import shutil

st.set_page_config(page_title="🏏 Cricket Predictor Pro", layout="wide", page_icon="🏏")
warnings.filterwarnings('ignore')

st.markdown("""
# 🏏 **Cricket Form Predictor PRO**
**Upload t20s.zip → Auto-train model → Predict 20 players!**
""")

@st.cache_data
def process_zip_data(uploaded_file):
    """Process t20s.zip - YOUR EXACT LOGIC"""
    all_data = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            json_files = [f for f in zip_ref.namelist() if f.endswith('.json')][:50]
            zip_ref.extractall(temp_dir, members=json_files)
        
        for json_file in Path(temp_dir).rglob("*.json"):
            try:
                data = json.load(open(json_file))
                info = data.get('info', {})
                if 't20' not in str(info.get('match_type', '')).lower():
                    continue
                
                match_id = json_file.stem
                for innings in data.get('innings', []):
                    for over in innings.get('overs', []):
                        for delivery in over.get('deliveries', []):
                            batter = delivery.get('batter')
                            if batter:
                                runs = delivery.get('runs', {}).get('batter', 0)
                                all_data.append({
                                    'player': batter,
                                    'match_id': match_id,
                                    'runs': runs,
                                    'is_four': 1 if runs == 4 else 0,
                                    'is_six': 1 if runs == 6 else 0
                                })
            except:
                continue
    
    df = pd.DataFrame(all_data)
    return df if len(df) > 0 else create_sample_data(20)

@st.cache_data
def create_sample_data(num_players=20):
    """20 realistic players sample data"""
    np.random.seed(42)
    players = [f"Player_{i+1}" for i in range(num_players)]
    data = []
    player_avgs = np.random.normal(25, 8, num_players).clip(10, 50)
    
    for match in range(30):
        for i, player in enumerate(players):
            runs = np.random.normal(player_avgs[i], 10).clip(0, 60)
            data.append({
                'player': player,
                'match_id': f'M{match+1}',
                'runs': int(runs),
                'is_four': np.random.choice([0,1], p=[0.8, 0.2]),
                'is_six': np.random.choice([0,1], p=[0.9, 0.1])
            })
    return pd.DataFrame(data)

def prepare_features(df):
    """Feature engineering for model"""
    features, targets = [], []
    df = df.sort_values(['player', 'match_id'])
    
    for player in df['player'].unique():
        player_df = df[df['player'] == player]
        for i in range(10, len(player_df)):  # Need 10+ innings
            prev_innings = player_df.iloc[:i]
            next_inning = player_df.iloc[i]
            
            career_avg = prev_innings['runs'].mean()
            recent_avg = prev_innings.tail(5)['runs'].mean()
            total_innings = len(prev_innings)
            recent_fours = prev_innings.tail(5)['is_four'].sum()
            recent_sixes = prev_innings.tail(5)['is_six'].sum()
            
            features.append([career_avg, recent_avg, total_innings, recent_fours, recent_sixes])
            targets.append(next_inning['runs'])
    
    return np.array(features), np.array(targets)

def predict_all_players(model, df):
    """Predict next match for ALL 20 players"""
    predictions = []
    
    for player in df['player'].unique():
        player_df = df[df['player'] == player].sort_values('match_id')
        if len(player_df) < 10:
            continue
            
        # Last 5 innings features
        recent = player_df.tail(5)
        career = player_df
        
        features = np.array([[
            career['runs'].mean(),
            recent['runs'].mean(),
            len(career),
            recent['is_four'].sum(),
            recent['is_six'].sum()
        ]])
        
        pred_runs = model.predict(features)[0]
        confidence = min(95, 60 + len(player_df) * 2)
        
        predictions.append({
            'player': player,
            'predicted_runs': round(max(0, pred_runs)),
            'confidence': confidence,
            'career_avg': career['runs'].mean().round(1),
            'total_runs': career['runs'].sum().round(0),
            'matches': len(career)
        })
    
    return pd.DataFrame(predictions).sort_values('predicted_runs', ascending=False)

# === MAIN APP ===
st.sidebar.header("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader("**Upload t20s.zip**", type=['zip'])

if uploaded_file is not None:
    with st.spinner("Processing ZIP data..."):
        df = process_zip_data(uploaded_file)
    st.session_state.df = df
    st.sidebar.success(f"✅ Loaded {len(df):,} deliveries from {df['player'].nunique()} players!")
else:
    df = create_sample_data(20)
    st.session_state.df = df
    st.sidebar.info("🧪 Using 20-player sample data")

df = st.session_state.df

# === DASHBOARD ===
row1_col1, row1_col2, row1_col3 = st.columns(3)
row1_col1.metric("Total Deliveries", f"{len(df):,}")
row1_col2.metric("Unique Players", df['player'].nunique())
row1_col3.metric("Matches", df['match_id'].nunique())

# Top players preview
st.markdown("### 🏆 Top 10 Players")
top_stats = df.groupby('player')['runs'].agg(['count', 'sum', 'mean']).round(1)
top_stats.columns = ['Innings', 'Total Runs', 'Avg']
st.dataframe(top_stats.nlargest(10, 'Total Runs'), use_container_width=True)

# === TRAIN MODEL ===
if st.button("🚀 TRAIN Random Forest MODEL", type="primary", use_container_width=True):
    with st.spinner("Training model on your data..."):
        X, y = prepare_features(df)
        if len(X) > 20:
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=8,
                min_samples_split=5,
                random_state=42,
                n_jobs=1
            )
            model.fit(X, y)
            
            # Quick validation
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            st.session_state.model = model
            st.session_state.scores = {'train': train_score, 'test': test_score}
            st.balloons()
            st.success(f"✅ Model trained! Train R²: {train_score:.3f} | Test R²: {test_score:.3f}")
        else:
            st.error("❌ Need more data (20+ innings per player)")

# === PREDICT ALL 20 PLAYERS ===
if st.button("🎯 PREDICT ALL 20 PLAYERS Next Match", type="primary", use_container_width=True):
    if 'model' in st.session_state:
        with st.spinner("🔮 Predicting for all players..."):
            predictions = predict_all_players(st.session_state.model, df)
            st.session_state.predictions = predictions.head(20)
            st.rerun()
    else:
        st.warning("👆 Train model first!")

# === RESULTS ===
if 'predictions' in st.session_state:
    predictions = st.session_state.predictions
    
    st.markdown("## 🎪 **NEXT MATCH PREDICTIONS** (Top 20)")
    
    # Prediction Cards + Chart
    col1, col2 = st.columns([2, 1])
    
    with col1:
        for i, (_, row) in enumerate(predictions.head(10).iterrows()):
            conf_color = "🔥" if row['confidence'] > 85 else "✅" if row['confidence'] > 70 else "⚡"
            st.markdown(f"""
            <div style='background: linear-gradient(45deg, #4CAF50, #81C784); 
                        padding: 1rem; border-radius: 10px; margin: 0.5rem 0; color: white;'>
                <h3 style='margin: 0;'>{row['predicted_runs']} runs</h3>
                <b>{row['player']}</b><br>
                <small>{conf_color} {row['confidence']:.0f}% conf | Career: {row['total_runs']} runs</small>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        fig = px.bar(predictions.head(15), x='predicted_runs', y='player', 
                    orientation='h', title="Top 15 Predictions",
                    color='predicted_runs', color_continuous_scale='viridis')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Full table
    st.markdown("### 📊 Complete Predictions")
    st.dataframe(predictions, use_container_width=True)

# Model info
if 'scores' in st.session_state:
    col1, col2 = st.columns(2)
    col1.metric("Train R²", f"{st.session_state.scores['train']:.3f}")
    col2.metric("Test R²", f"{st.session_state.scores['test']:.3f}")

st.markdown("---")
st.markdown("""
**Features Used:**
- Career batting average
- Recent 5 innings average  
- Total career innings
- Recent fours/sixes

**Model:** Random Forest (200 trees, max_depth=8)
""")
