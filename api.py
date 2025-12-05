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

# GitHub URLs
DATA_ZIP_URL = "https://github.com/aksmamg014/Cricket_Form_Analyzer/blob/main/t20s.zip?raw=true"
MODEL_ZIP_URL = "https://github.com/aksmamg014/Cricket_Form_Analyzer/blob/main/models.zip?raw=true"
MODEL_DIR = "models/rf_cricket_score"
DATA_DIR = "t20s_data"

# ==========================================
# SETUP FUNCTION
# ==========================================

def initialize_system():
    """
    Called ONCE to download, extract, and load everything.
    Uses session_state to store results.
    """
    st.session_state.logs = []
    
    # --- Helper to download and log ---
    def download_and_extract(url, target_dir, desc):
        if os.path.exists(target_dir) and len(os.listdir(target_dir)) > 0:
            st.session_state.logs.append(f"✅ {desc} found locally.")
            return True
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(target_dir if desc == "Data" else ".")
            st.session_state.logs.append(f"✅ {desc} downloaded.")
            return True
        except Exception as e:
            st.session_state.logs.append(f"❌ Failed to download {desc}: {e}")
            return False

    # 1. Get Data
    if not download_and_extract(DATA_ZIP_URL, DATA_DIR, "Data"):
        return

    # 2. Get Model
    if not download_and_extract(MODEL_ZIP_URL, ".", "Model"):
        return
        
    # 3. Load Model and Features
    try:
        if os.path.exists(MODEL_DIR):
            files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".joblib") and "model" in f]
            if not files:
                st.session_state.logs.append("❌ No model .joblib file found.")
                return

            payload = joblib.load(os.path.join(MODEL_DIR, sorted(files)[-1]))
            st.session_state.model = payload['model']
            st.session_state.features = payload['feature_names'].tolist()
            st.session_state.logs.append("✅ Model loaded.")
        else:
            st.session_state.logs.append("❌ Model directory not found after unzip.")
    except Exception as e:
        st.session_state.logs.append(f"❌ Failed to load model: {e}")

    # 4. Load Player Data
   import requests
import zipfile
import io
import pandas as pd
import os

# 1. Define URL (Must use ?raw=true)
url = "https://github.com/aksmamg014/Cricket_Form_Analyzer/blob/main/t20s.zip?raw=true"

# 2. Fetch content
r = requests.get(url)
r.raise_for_status()

# 3. Extract to a folder
with zipfile.ZipFile(io.BytesIO(r.content)) as z:
    z.extractall("t20s_data")

# 4. Find and Read CSV
# This looks inside the extracted folder to find the actual CSV file
csv_files = [f for f in os.listdir("t20s_data") if f.endswith(".csv")]
if csv_files:
    df = pd.read_csv(f"t20s_data/{csv_files[0]}")
    print(f"Loaded {len(df)} rows")

# ==========================================
# MAIN UI
# ==========================================

st.title("🏏 Cricket Player Score Predictor")
st.markdown("---")

# Initialize session state if it doesn't exist
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.player_data = None
    st.session_state.features = None
    st.session_state.logs = []

# --- Section 1: Initialization ---
if st.session_state.model is None:
    st.subheader("System Initialization")
    st.warning("System is not ready. Please initialize to download models and data.")
    
    if st.button("🚀 Initialize System", type="primary"):
        with st.spinner("Downloading and setting up... This may take a moment."):
            initialize_system()
        st.rerun()
    
    # Show logs if they exist
    if st.session_state.logs:
        st.subheader("Logs")
        st.code("\n".join(st.session_state.logs))

# --- Section 2: Main App ---
else:
    st.success("✅ System Ready")
    
    # Find player column
    player_col = [c for c in st.session_state.player_data.columns if c.lower() in ['player', 'batsman']][0]
    players = sorted(st.session_state.player_data[player_col].astype(str).unique())

    # Player Selection UI
    selected_player = st.selectbox("Select Player", players)
    
    if selected_player:
        # Prediction Logic
        if st.button(f"Predict Score for {selected_player}"):
            # THIS IS A DUMMY FEATURE PREPARATION
            # YOU MUST REPLACE THIS with your actual logic to calculate features
            # for the selected player from their historical data.
            dummy_features = pd.DataFrame([0]*len(st.session_state.features), index=st.session_state.features).T
            
            prediction = st.session_state.model.predict(dummy_features)[0]
            st.metric(label="Predicted Next Match Score", value=f"{int(prediction)} Runs")

# --- Section 3: Debug Log Expander (at the bottom) ---
if st.session_state.logs:
    with st.expander("Show Initialization Logs"):
        st.code("\n".join(st.session_state.logs))

