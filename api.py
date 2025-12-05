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
   ef load_model_from_github():
    """
    Downloads 'models.zip' from GitHub, extracts it, and loads the model.
    Returns: model, feature_names (list)
    """
    # 1. GitHub URL (Must use ?raw=true)
    MODEL_ZIP_URL = "https://github.com/aksmamg014/Cricket_Form_Analyzer/blob/main/models.zip?raw=true"
    EXTRACT_DIR = "models_extracted"
    
    status_msg = []
    
    try:
        # 2. Download Zip
        if not os.path.exists(EXTRACT_DIR):
            status_msg.append("📥 Downloading model zip...")
            response = requests.get(MODEL_ZIP_URL, timeout=60)
            response.raise_for_status()
            
            # 3. Extract Zip
            status_msg.append("📦 Extracting model...")
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                z.extractall(EXTRACT_DIR)
        else:
            status_msg.append("✅ Model folder exists (Using cached).")

        # 4. Find .joblib File Recursively
        # This logic finds the file even if it's hidden in models_extracted/models/rf_score/...
        model_path = None
        for root, dirs, files in os.walk(EXTRACT_DIR):
            for file in files:
                # Look for file starting with 'rf_model' and ending with '.joblib'
                if file.startswith("rf_model") and file.endswith(".joblib"):
                    model_path = os.path.join(root, file)
                    # If multiple exist, we want the latest one (by name usually implies date)
                    # This loop naturally keeps finding files, so we can sort them later if needed
                    # But taking the first valid one is usually enough if you clean your zip.
        
        if not model_path:
            st.error("❌ No model file (rf_model*.joblib) found inside the zip!")
            return None, None

        # 5. Load Model using Joblib
        status_msg.append(f"🚀 Loading: {os.path.basename(model_path)}")
        payload = joblib.load(model_path)
        
        # 6. Parse Payload
        model = None
        features = []
        
        if isinstance(payload, dict) and 'model' in payload:
            model = payload['model']
            raw_features = payload.get('feature_names')
            
            # Fix feature format
            if raw_features is not None:
                if hasattr(raw_features, 'tolist'):
                    features = raw_features.tolist()
                elif hasattr(raw_features, 'columns'):
                    features = raw_features.columns.tolist()
                else:
                    features = list(raw_features)
        else:
            model = payload # Fallback if just the model object was saved
            
        return model, features

    except Exception as e:
        st.error(f"❌ Model Load Failed: {str(e)}")
        return None, None

# --- How to use in your App ---
# (Only run this when you need the model)

if 'model' not in st.session_state:
    model, features = load_model_from_github()
    
    if model:
        st.session_state.model = model
        st.session_state.features = features
        st.success("Model loaded successfully!")
    else:
        st.stop() # Stop app if model fails

    # 4. Load Player Data
  def load_t20_data_from_github():
    """
    Downloads 't20s.zip' from GitHub, extracts it, and loads the CSV.
    Returns: pd.DataFrame
    """
    # 1. GitHub URL (Must use ?raw=true)
    DATA_ZIP_URL = "https://github.com/aksmamg014/Cricket_Form_Analyzer/blob/main/t20s.zip?raw=true"
    EXTRACT_DIR = "t20s_data_extracted"
    
    try:
        # 2. Download & Extract (Only if not already there)
        if not os.path.exists(EXTRACT_DIR):
            st.info("📥 Downloading T20 data...")
            response = requests.get(DATA_ZIP_URL, timeout=60)
            response.raise_for_status()
            
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                z.extractall(EXTRACT_DIR)
        else:
            # Optional: Check if folder is empty, if so, re-download
            if not os.listdir(EXTRACT_DIR):
                 st.warning("⚠️ Data folder empty, re-downloading...")
                 # (You could add re-download logic here if needed)

        # 3. Find the CSV File Recursively
        # This finds the CSV even if it's inside t20s_data_extracted/t20s/data.csv
        csv_path = None
        for root, dirs, files in os.walk(EXTRACT_DIR):
            for file in files:
                if file.endswith(".csv"):
                    csv_path = os.path.join(root, file)
                    break # Found one!
            if csv_path: break
            
        if not csv_path:
            st.error("❌ No CSV file found inside the zip!")
            return None

        # 4. Load into Pandas
        st.info(f"📊 Loading data from: {os.path.basename(csv_path)}")
        
        # Optimization: Use low_memory=False to prevent type warnings on large files
        df = pd.read_csv(csv_path, low_memory=False)
        
        st.success(f"✅ Data Loaded: {len(df)} records found.")
        return df

    except Exception as e:
        st.error(f"❌ Data Load Failed: {str(e)}")
        return None

# --- Usage in your App ---

if 'player_data' not in st.session_state:
    df = load_t20_data_from_github()
    if df is not None:
        st.session_state.player_data = df
    else:
        st.stop() # Stop if data fails

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


