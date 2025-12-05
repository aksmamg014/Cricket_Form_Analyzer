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
st.set_page_config(page_title="Cricket Score Predictor", page_icon="🏏", layout="wide")

# -------------------------------------------
# 1. Define URLs and File Names
# -------------------------------------------
# URL for the DATA zip (t20s.zip)
DATA_ZIP_URL = "https://github.com/aksmamg014/Cricket_Form_Analyzer/blob/main/t20s.zip?raw=true"

# Name of the MODEL zip (must exist in your repo root)
MODEL_ZIP_FILE = "models.zip" 

# Directory where models are extracted
MODEL_DIR = "models/rf_cricket_score"

# Directory where data is extracted
DATA_DIR = "t20s_data" # We'll extract data here

# ==========================================
# CACHED SETUP FUNCTIONS (Run Automatically)
# ==========================================

@st.cache_resource
def setup_model_files():
    """
    Ensures model files are present. Unzips 'models.zip' if needed.
    """
    if os.path.exists(MODEL_DIR) and len(os.listdir(MODEL_DIR)) > 0:
        return True
        
    if os.path.exists(MODEL_ZIP_FILE):
        try:
            with zipfile.ZipFile(MODEL_ZIP_FILE, 'r') as zip_ref:
                zip_ref.extractall(".") 
            if os.path.exists(MODEL_DIR) and len(os.listdir(MODEL_DIR)) > 0:
                return True
        except Exception as e:
            st.error(f"Failed to unzip model: {e}")
            return False
    return False

@st.cache_resource
def download_data_on_startup():
    """
    Automatically downloads and extracts the data zip from GitHub 
    if the data folder doesn't exist.
    """
    # Check if data is already there
    if os.path.exists(DATA_DIR) and len(os.listdir(DATA_DIR)) > 0:
        return f"✅ Data ready in `{DATA_DIR}` (Cached)"

    # Download
    try:
        response = requests.get(DATA_ZIP_URL)
        response.raise_for_status()
        
        # Unzip
        z = zipfile.ZipFile(io.BytesIO(response.content))
        z.extractall(DATA_DIR) # Extract into our specific data folder
        
        return f"✅ Data downloaded & extracted to `{DATA_DIR}`"
    except Exception as e:
        return f"❌ Data download failed: {e}"

@st.cache_resource
def load_model_artifacts(model_dir):
    """Loads model, features, and metadata safely."""
    if not setup_model_files():
        return None, None, {}

    if not os.path.exists(model_dir):
        return None, None, {}

    try:
        files = [f for f in os.listdir(model_dir) if f.startswith("rf_model_") and f.endswith(".joblib")]
    except FileNotFoundError:
        return None, None, {}
        
    if not files:
        return None, None, {}

    latest_file = sorted(files)[-1]
    model_path = os.path.join(model_dir, latest_file)
    
    try:
        payload = joblib.load(model_path)
        
        model_obj = None
        features = None
        meta = {}

        if isinstance(payload, dict) and 'model' in payload:
            model_obj = payload['model']
            features = payload.get('feature_names')
            meta = payload.get('metadata', {})
        else:
            model_obj = payload
        
        # Fix DataFrame ambiguity
        if features is not None:
            if hasattr(features, 'tolist'):
                features = features.tolist()
            elif hasattr(features, 'columns'):
                features = features.columns.tolist()
        
        return model_obj, features, meta

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, {}

@st.cache_resource
def load_feature_names_fallback(model_dir):
    path = os.path.join(model_dir, "feature_names.joblib")
    if os.path.exists(path):
        names = joblib.load(path)
        if hasattr(names, 'tolist'):
            return names.tolist()
        return names
    return None

# ==========================================
# MAIN UI
# ==========================================

st.title("🏏 Cricket Score Predictor")

# --- 1. Auto-Run Setup ---
# This runs immediately when the script starts
data_status = download_data_on_startup()
model, meta_features, metadata = load_model_artifacts(MODEL_DIR)

if (meta_features is None or len(meta_features) == 0) and model is not None:
    meta_features = load_feature_names_fallback(MODEL_DIR)

# --- 2. Sidebar Status ---
with st.sidebar:
    st.header("⚙️ System Status")
    
    # Data Status
    if "✅" in data_status:
        st.success(data_status)
    else:
        st.error(data_status)
        
    # Model Status
    if model is not None:
        st.success(f"✅ Model Loaded")
        if metadata.get('timestamp'):
            st.caption(f"Version: {metadata.get('timestamp')}")
            st.metric("Model MAE", f"{metadata.get('cv_mae_mean', 0):.2f}")
    else:
        st.error("❌ Model Missing")

# --- 3. Prediction Interface ---
if model is not None and meta_features is not None:
    st.subheader("🔮 Predict Score")
    
    with st.form("prediction_form"):
        st.info("Enter match conditions below:")
        
        cols = st.columns(3)
        input_data = {}
        
        for i, feature in enumerate(meta_features):
            with cols[i % 3]:
                # Smart Defaults
                val = 0.0
                min_v = 0.0
                max_v = None
                step = 1.0
                
                feature_lower = str(feature).lower()
                
                if 'runs' in feature_lower or 'score' in feature_lower:
                    val = 160.0
                elif 'wickets' in feature_lower:
                    val = 2.0
                    max_v = 10.0
                elif 'overs' in feature_lower:
                    val = 10.0
                    max_v = 50.0
                    step = 0.1
                elif 'rate' in feature_lower: # Run rate, etc.
                    val = 8.0
                    step = 0.1
                
                input_data[feature] = st.number_input(
                    label=feature, 
                    min_value=min_v, 
                    max_value=max_v, 
                    value=val,
                    step=step
                )

        submitted = st.form_submit_button("🚀 Run Prediction")

    if submitted:
        try:
            df_input = pd.DataFrame([input_data])
            prediction = model.predict(df_input)[0]
            
            st.markdown("---")
            st.markdown(f"### 🎯 Predicted Score: **{int(prediction)}**")
            
            # Optional: Show input summary
            with st.expander("See Input Summary"):
                st.dataframe(df_input)
                
        except Exception as e:
            st.error(f"Prediction Error: {e}")

else:
    st.warning("⚠️ System initializing... if this persists, check logs.")







