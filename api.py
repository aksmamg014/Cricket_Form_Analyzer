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
DATA_ZIP_URL = "https://github.com/aksmamg014/Cricket_Form_Analyzer/blob/main/t20s.zip?raw=true"
MODEL_ZIP_URL = "https://github.com/aksmamg014/Cricket_Form_Analyzer/blob/main/models.zip?raw=true"

# Directories for extraction
MODEL_EXTRACT_DIR = "models_extracted"
DATA_EXTRACT_DIR = "t20s_data_extracted"

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
    logs = []

    # 1. Download Data
    ok, msg = download_and_extract(DATA_ZIP_URL, DATA_EXTRACT_DIR, "Data")
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

    # 4. Load Data
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

    # Success
    st.session_state.logs = logs
    st.session_state.system_ready = True
        
        # Handle Payload
        if isinstance(payload, dict) and 'model' in payload:
            st.session_state.model = payload['model']
            raw_features = payload.get('feature_names')
            
            # Fix Feature Names
            if raw_features is not None:
                if hasattr(raw_features, 'tolist'):
                    st.session_state.features = raw_features.tolist()
                elif hasattr(raw_features, 'columns'):
                    st.session_state.features = raw_features.columns.tolist()
                else:
                    st.session_state.features = list(raw_features)
            else:
                st.session_state.features = []
        else:
            st.session_state.model = payload
            st.session_state.features = [] # Assuming no features if raw model
            
        logs.append("✅ Model loaded into memory.")

    except Exception as e:
        logs.append(f"❌ Model Load Error: {str(e)}")
        st.session_state.logs = logs
        return

    # 4. Load Data
       # ... inside initialize_system > 4. Load Data ...
    try:
        # Find JSON file recursively
        json_path = recursive_find_file(DATA_EXTRACT_DIR, ".json")
        
        if not json_path:
            logs.append("❌ No JSON file found in data zip.")
            st.session_state.logs = logs
            return

        logs.append(f"📊 Loading data from: {os.path.basename(json_path)}")
        
        # READ JSON
        # Note: 'orient' depends on your JSON structure. 
        # Common ones: 'records', 'split', 'index'. 
        # If unsure, try without arguments first.
        try:
            st.session_state.player_data = pd.read_json(json_path)
        except ValueError:
             # Fallback: sometimes JSONs are line-delimited (NDJSON)
            st.session_state.player_data = pd.read_json(json_path, lines=True)
            
        logs.append(f"✅ Data loaded: {len(st.session_state.player_data)} rows.")

    except Exception as e:
        logs.append(f"❌ Data Load Error: {str(e)}")
        st.session_state.logs = logs
        return


    # Success
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
    st.info("Click the button below to download models and data from GitHub.")
    
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
    st.success("✅ System is Ready")
    
    # 1. Player Selection
    df = st.session_state.player_data
    
    # Try to find player name column
    possible_cols = [c for c in df.columns if c.lower() in ['player', 'batsman', 'batter', 'striker', 'player_name']]
    player_col = possible_cols[0] if possible_cols else None
    
    if player_col:
        players = sorted(df[player_col].astype(str).unique())
        selected_player = st.selectbox("Select Player:", players)
        
        if selected_player:
            # Show minimal stats
            p_data = df[df[player_col] == selected_player]
            st.write(f"**Matches Played:** {len(p_data)}")
            
            # 2. Prediction Button
            if st.button(f"Predict Next Score for {selected_player}"):
                if st.session_state.model:
                    # --- FEATURE PREPARATION (CRITICAL) ---
                    # currently sending zeros just to prove model works
                    # You need to replace this logic with your actual feature calculation!
                    try:
                        dummy_features = pd.DataFrame([0]*len(st.session_state.features), columns=st.session_state.features)
                        prediction = st.session_state.model.predict(dummy_features)[0]
                        
                        st.markdown("### 🎯 Prediction Result")
                        st.metric(label="Predicted Score", value=f"{int(prediction)} Runs")
                    except Exception as e:
                        st.error(f"Prediction Error: {e}")
                else:
                    st.error("Model object is missing.")
    else:
        st.error(f"Could not find a 'player' column in the CSV. Columns found: {list(df.columns)}")
        
    # Reset Button (for debugging)
    if st.button("Reset System"):
        st.session_state.clear()
        st.rerun()




