import streamlit as st
import pandas as pd
import joblib
import os
import requests
import zipfile
import io

# ==========================================
# CONFIGURATION
# ==========================================
st.set_page_config(page_title="Cricket Player Score Predictor", page_icon="🏏", layout="wide")

# GitHub URLs (Must end in ?raw=true)
DATA_CSV_URL = "https://github.com/aksmamg014/Cricket_Form_Analyzer/raw/main/t20_data.csv"
MODEL_ZIP_URL = "https://github.com/aksmamg014/Cricket_Form_Analyzer/blob/main/models.zip?raw=true"

# Directories for extraction
MODEL_EXTRACT_DIR = "models_extracted"
DATA_CSV_PATH = "t20_data.csv"  # Local path after download

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def recursive_find_file(directory, endswith):
    """Helper to find a file recursively in a directory."""
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(endswith):
                return os.path.join(root, file)
    return None

def download_csv(url, local_path, desc):
    """Downloads CSV if it doesn't exist locally."""
    if os.path.exists(local_path):
        return True, f"✅ {desc} found locally."
    
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        with open(local_path, 'wb') as f:
            f.write(response.content)
        return True, f"✅ {desc} downloaded."
    except Exception as e:
        return False, f"❌ Failed to download {desc}: {str(e)}"

def download_and_extract(url, target_dir, desc):
    """Downloads and extracts zip if target_dir doesn't exist."""
    if os.path.exists(target_dir) and len(os.listdir(target_dir)) > 0:
        return True, f"✅ {desc} found locally."
    
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(target_dir)
        return True, f"✅ {desc} downloaded & extracted."
    except Exception as e:
        return False, f"❌ Failed to download {desc}: {str(e)}"

# ==========================================
# INITIALIZATION LOGIC
# ==========================================

def initialize_system():
    # Initialize session state variables if they don't exist
    if 'logs' not in st.session_state:
        st.session_state.logs = []
        
    logs = []

    # 1. Download Data CSV
    ok, msg = download_csv(DATA_CSV_URL, DATA_CSV_PATH, "t20_data.csv")
    logs.append(msg)
    if not ok:
        st.session_state.logs = logs
        return

    # 2. Download Model
    ok, msg = download_and_extract(MODEL_ZIP_URL, MODEL_EXTRACT_DIR, "Model")
    logs.append(msg)
    if not ok:
        st.session_state.logs = logs
        return

    # 3. Load Model
    try:
        model_path = recursive_find_file(MODEL_EXTRACT_DIR, ".joblib")
        if not model_path:
            logs.append("❌ No .joblib file found in model zip.")
            st.session_state.logs = logs
            return

        logs.append(f"📦 Loading model from: {os.path.basename(model_path)}")
        payload = joblib.load(model_path)

        if isinstance(payload, dict) and 'model' in payload:
            st.session_state.model = payload['model']
            raw_features = payload.get('feature_names', [])
            if hasattr(raw_features, 'tolist'):
                st.session_state.features = raw_features.tolist()
            elif hasattr(raw_features, 'columns'):
                st.session_state.features = raw_features.columns.tolist()
            else:
                st.session_state.features = list(raw_features)
        else:
            st.session_state.model = payload
            st.session_state.features = []

        logs.append("✅ Model loaded into memory.")
    
    except Exception as e:
        logs.append(f"❌ Model Load Error: {str(e)}")
        st.session_state.logs = logs
        return

    # 4. Load CSV Data
    try:
        if not os.path.exists(DATA_CSV_PATH):
            logs.append("❌ CSV file missing after download.")
            st.session_state.logs = logs
            return

        logs.append(f"📊 Loading data from: {DATA_CSV_PATH}")
        
        # READ CSV with error handling
        try:
            st.session_state.player_data = pd.read_csv(DATA_CSV_PATH)
        except pd.errors.ParserError as e:
            logs.append(f"⚠️ Comma separator failed, trying tab separator...")
            st.session_state.player_data = pd.read_csv(DATA_CSV_PATH, sep='\t')
        except UnicodeDecodeError:
            logs.append(f"⚠️ UTF-8 encoding failed, trying latin-1...")
            st.session_state.player_data = pd.read_csv(DATA_CSV_PATH, encoding='latin-1')
            
        logs.append(f"✅ Data loaded: {len(st.session_state.player_data)} rows, {len(st.session_state.player_data.columns)} columns.")

    except Exception as e:
        logs.append(f"❌ Data Load Error: {str(e)}")
        st.session_state.logs = logs
        return

    # 5. Final Success
    st.session_state.logs = logs
    st.session_state.system_ready = True

# ==========================================
# MAIN APP UI
# ==========================================

st.title("🏏 Cricket Player Score Predictor")
st.markdown("---")

# Initialize Session State Variables
if 'system_ready' not in st.session_state:
    st.session_state.system_ready = False
    st.session_state.logs = []
    st.session_state.model = None
    st.session_state.player_data = None
    st.session_state.features = []

# --- VIEW 1: INITIALIZATION SCREEN ---
if not st.session_state.system_ready:
    st.subheader("System Initialization")
    st.info("Click the button below to download t20_data.csv and models from GitHub.")
    
    if st.button("🚀 Initialize System", type="primary"):
        with st.status("Initializing...", expanded=True) as status:
            initialize_system()
            # If successful, rerun to show main app
            if st.session_state.system_ready:
                status.update(label="System Ready!", state="complete", expanded=False)
                st.rerun()
            else:
                status.update(label="Initialization Failed", state="error")
    
    # Show logs
    if st.session_state.logs:
        with st.expander("View Logs", expanded=True):
            for log in st.session_state.logs:
                st.write(log)

# --- VIEW 2: MAIN PREDICTION SCREEN ---
# --- VIEW 2: MAIN PREDICTION SCREEN ---
else:
    st.success("✅ System Ready! Model expects: career_avg, player_id, prev_runs, roll_avg_3, roll_avg_5")
    
    df = st.session_state.player_data
    
    if df is not None and not df.empty:
        # Dashboard metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Matches", len(df))
        with col2:
            player_col = next((c for c in df.columns if c.lower() in ['player', 'batsman', 'player_name']), None)
            total_players = len(df[player_col].unique()) if player_col else 0
            st.metric("Total Players", total_players)
        
        # Data preview
        with st.expander("📋 Dataset Preview"):
            st.write(f"**Shape:** {df.shape}")
            st.write(f"**Columns:** {list(df.columns)}")
            st.dataframe(df.head())
        
        # Player selection
        player_col = next((c for c in df.columns if c.lower() in ['player', 'batsman', 'player_name']), None)
        selected_player = None
        
        if player_col:
            players = sorted(df[player_col].unique())[:100]
            selected_player = st.selectbox("👨‍🦰 Select Player", players)
            
            if selected_player:
                p_data = df[df[player_col] == selected_player].copy()
                
                # Player header
                st.markdown(f"### 📈 {selected_player}'s Stats")
                col1, col2, col3 = st.columns(3)
                
                # Find score column
                score_col = next((c for c in df.columns if 'score' in c.lower() or 'runs' in c.lower()), None)
                
                with col1:
                    st.metric("Matches", len(p_data))
                
                if score_col:
                    p_scores = p_data[score_col].dropna()
                    with col2:
                        st.metric("Career Avg", f"{p_scores.mean():.1f}" if len(p_scores) > 0 else "N/A")
                    with col3:
                        st.metric("Best Score", f"{int(p_scores.max())}" if len(p_scores) > 0 else "N/A")
                else:
                    with col2:
                        st.metric("Avg Score", "No score column")
                    with col3:
                        st.metric("Best", "N/A")
                
                # 🎯 PREDICTION BUTTON
                if st.button(f"🎯 Predict {selected_player}'s Next Score", use_container_width=True):
                    if st.session_state.model:
                        try:
                            p_scores = p_data[score_col].dropna() if score_col else pd.Series([])
                            
                            feature_data = {
                                'career_avg': [p_scores.mean() if len(p_scores) > 0 else 25.0],
                                'player_id': [hash(selected_player) % 10000],
                                'prev_runs': [p_scores.iloc[-1] if len(p_scores) > 0 else 20.0],
                                'roll_avg_3': [p_scores.tail(3).mean() if len(p_scores) >= 3 else 
                                             (p_scores.mean() if len(p_scores) > 0 else 22.0)],
                                'roll_avg_5': [p_scores.tail(5).mean() if len(p_scores) >= 5 else 
                                             (p_scores.mean() if len(p_scores) > 0 else 20.0)]
                            }
                            
                            input_features = pd.DataFrame(feature_data)
                            
                            st.success("✅ Generated player-specific features!")
                            st.dataframe(input_features)
                            
                            prediction = st.session_state.model.predict(input_features)[0]
                            
                            st.markdown("### 🎯 Prediction Result")
                            st.metric("Predicted Next Score", f"{int(prediction)} Runs")
                            
                            if len(p_scores) >= 5:
                                confidence = p_scores.tail(5).std()
                                st.metric("Confidence (±)", f"{confidence:.0f} runs")
                            
                            if hasattr(st.session_state.model, 'feature_importances_'):
                                st.markdown("### 📊 Feature Importance")
                                importance_df = pd.DataFrame({
                                    'Feature': ['career_avg', 'player_id', 'prev_runs', 'roll_avg_3', 'roll_avg_5'],
                                    'Importance': st.session_state.model.feature_importances_
                                }).sort_values('Importance', ascending=False)
                                st.bar_chart(importance_df.set_index('Feature'))
                                st.dataframe(importance_df)
                        
                        except Exception as e:
                            st.error(f"❌ Prediction Error: {str(e)}")
                            st.info(f"Debug - Features: {st.session_state.features}")
                    else:
                        st.error("❌ Model not loaded.")
            else:
                st.info("👆 Select a player above")
        else:
            st.error("❌ No player column found")
            st.write("Columns:", list(df.columns))
    
    st.divider()
    if st.button("🔄 Reset System", type="secondary"):
        st.session_state.clear()
        st.rerun()


