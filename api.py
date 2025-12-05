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
else:
    st.success("✅ System is Ready!")
    
    df = st.session_state.player_data
    
    if df is not None and not df.empty:
        # Data preview
        with st.expander("📋 Dataset Preview"):
            st.write(f"**Shape:** {df.shape}")
            st.dataframe(df.head())
        
        # Model features display
        st.info(f"**Model expects these 5 features:** career_avg, player_id, prev_runs, roll_avg_3, roll_avg_5")
        
        # Player selection
        possible_cols = [c for c in df.columns if c.lower() in ['player', 'batsman', 'batter', 'striker', 'player_name']]
        player_col = possible_cols[0] if possible_cols else None
        
        if player_col:
            players = sorted(df[player_col].astype(str).unique())[:50]  # Limit for UI
            selected_player = st.selectbox("Select Player:", players)
            
            if selected_player:
                p_data = df[df[player_col] == selected_player]
                
                # Player stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Matches", len(p_data))
                
                # Find score column
                score_col = None
                for col in df.columns:
                    if 'score' in col.lower() or 'runs' in col.lower():
                        score_col = col
                        break
                
                if score_col:
                    with col2:
                        avg_score = p_data[score_col].mean()
                        st.metric("Career Avg", f"{avg_score:.1f}")
                    with col3:
                        last_score = p_data[score_col].iloc[-1] if len(p_data) > 0 else 0
                        st.metric("Last Score", f"{int(last_score)}")
                
                # # 2. COMPLETE PREDICTION BUTTON - Using REAL player data
if st.button(f"🎯 Predict Next Score for {selected_player}", use_container_width=True):
    if st.session_state.model:
        try:
            # Find score column first
            score_col = None
            for col in df.columns:
                if 'score' in col.lower() or 'runs' in col.lower():
                    score_col = col
                    break
            
            p_scores = p_data[score_col].dropna() if score_col and len(p_data) > 0 else pd.Series([])
            
            # ✅ EXACT FEATURES with REAL player data
            feature_data = {
                'career_avg': [p_scores.mean() if len(p_scores) > 0 else 25.0],
                'player_id': [hash(selected_player) % 10000],  # Unique numeric ID
                'prev_runs': [p_scores.iloc[-1] if len(p_scores) > 0 else 20.0],
                'roll_avg_3': [p_scores.tail(3).mean() if len(p_scores) >= 3 else p_scores.mean() if len(p_scores) > 0 else 22.0],
                'roll_avg_5': [p_scores.tail(5).mean() if len(p_scores) >= 5 else p_scores.mean() if len(p_scores) > 0 else 20.0]
            }
            
            input_features = pd.DataFrame(feature_data)
            
            st.success("✅ Generated player-specific features from match history!")
            st.dataframe(input_features, use_container_width=True)
            
            # Make prediction
            prediction = st.session_state.model.predict(input_features)[0]
            
            st.markdown("### 🎯 Prediction Result")
            st.metric(label="Predicted Next Match Score", value=f"{int(prediction)} Runs")
            
            # Show prediction confidence (std of last 5 matches)
            if len(p_scores) >= 5:
                confidence = p_scores.tail(5).std()
                st.metric("Prediction Confidence (±)", f"{confidence:.0f} runs")
            
            # Feature importance visualization
            if hasattr(st.session_state.model, 'feature_importances_'):
                st.markdown("### 📊 What Drives This Prediction?")
                importance_df = pd.DataFrame({
                    'Feature': ['career_avg', 'player_id', 'prev_runs', 'roll_avg_3', 'roll_avg_5'],
                    'Importance': st.session_state.model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                st.bar_chart(importance_df.set_index('Feature'))
                st.dataframe(importance_df)
            
        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")
            st.info("Debug info:")
            st.write(f"- Features available: {st.session_state.features}")
            st.write(f"- Score column found: {score_col}")
            st.write(f"- Player matches: {len(p_data)}")
    else:
        st.error("❌ Model not loaded.")




