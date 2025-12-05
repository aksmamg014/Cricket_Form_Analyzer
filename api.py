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

# Placeholder for your GitHub Raw URL (User must update this)
GITHUB_ZIP_URL = "https://github.com/aksmamg014/Cricket_Form_Analyzer/blob/main/t20s.zip"

# Directory where we want the model files
MODEL_DIR = "models/rf_cricket_score"
# Name of the zip file you pushed to GitHub
MODEL_ZIP_FILE = "models.zip" 

# ==========================================
# SETUP & CACHED FUNCTIONS
# ==========================================

@st.cache_resource
def setup_model_files():
    """
    Checks if model files exist. If not, tries to unzip 'model.zip'.
    Returns True if successful, False otherwise.
    """
    # Check if directory exists and has files
    if os.path.exists(MODEL_DIR) and os.listdir(MODEL_DIR):
        return True
        
    # If not, try to unzip
    if os.path.exists(MODEL_ZIP_FILE):
        try:
            with zipfile.ZipFile(MODEL_ZIP_FILE, 'r') as zip_ref:
                zip_ref.extractall(".") # Extracts to current directory, preserving paths
            return True
        except Exception as e:
            st.error(f"Failed to unzip model file: {e}")
            return False
    return False

@st.cache_resource
def load_latest_model(model_dir):
    """Loads the latest .joblib model from the directory."""
    
    # Ensure files are ready
    if not setup_model_files():
        return None, None, None

    if not os.path.exists(model_dir):
        return None, None, None

    # Find all model files
    files = [f for f in os.listdir(model_dir) if f.startswith("rf_model_") and f.endswith(".joblib")]
    if not files:
        return None, None, None

    # Sort by timestamp (latest last)
    latest_file = sorted(files)[-1]
    model_path = os.path.join(model_dir, latest_file)
    
    try:
        payload = joblib.load(model_path)
        if isinstance(payload, dict) and 'model' in payload:
            return payload['model'], payload.get('feature_names'), payload.get('metadata')
        else:
            return payload, None, {}
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

@st.cache_resource
def load_feature_names(model_dir):
    path = os.path.join(model_dir, "feature_names.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    return None

@st.cache_data
def download_and_extract_github_data(url):
    if "YOUR_USER" in url: 
        return None
    try:
        response = requests.get(url)
        response.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(response.content))
        extract_path = "temp_data"
        z.extractall(extract_path)
        return extract_path
    except Exception as e:
        st.error(f"Failed to download data from GitHub: {e}")
        return None

# ==========================================
# MAIN UI
# ==========================================

st.title("🏏 Cricket Score Predictor")
st.markdown(f"**Model:** Random Forest | **Backend:** Scikit-Learn")

# --- Sidebar: Settings ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Dynamic Model Loader
    model, meta_features, metadata = load_latest_model(MODEL_DIR)
    
    # Fallback to separate feature file
    if not meta_features and model:
        meta_features = load_feature_names(MODEL_DIR)
    
    if model:
        st.success(f"✅ Model Loaded!")
        if metadata and 'timestamp' in metadata:
            st.caption(f"Trained: {metadata['timestamp']}")
            st.metric("CV MAE", f"{metadata.get('cv_mae_mean', 0):.2f}")
    else:
        st.error("❌ Model not found. Please check 'model.zip' is in repo.")
    
    st.markdown("---")
    st.subheader("🔗 Data Source")
    github_url = st.text_input("GitHub Zip URL", value=GITHUB_ZIP_URL)
    
    if st.button("Download Data"):
        data_path = download_and_extract_github_data(github_url)
        if data_path:
            st.success(f"Data extracted to `{data_path}`")

# --- Main Prediction Area ---

if model and meta_features:
    st.subheader("🔮 Make a Prediction")
    with st.form("prediction_form"):
        st.markdown("### Input Features")
        cols = st.columns(3)
        input_data = {}
        
        for i, feature in enumerate(meta_features):
            with cols[i % 3]:
                if 'runs' in feature.lower() or 'score' in feature.lower():
                    input_data[feature] = st.number_input(feature, min_value=0, value=100)
                elif 'wickets' in feature.lower():
                    input_data[feature] = st.number_input(feature, min_value=0, max_value=10, value=2)
                elif 'overs' in feature.lower():
                    input_data[feature] = st.number_input(feature, min_value=0.0, max_value=50.0, value=10.0)
                else:
                    input_data[feature] = st.number_input(feature, value=0.0)

        submit = st.form_submit_button("🚀 Predict Score")

    if submit:
        df_input = pd.DataFrame([input_data])
        prediction = model.predict(df_input)[0]
        st.markdown("---")
        st.metric(label="Predicted Score", value=f"{prediction:.0f} Runs")
else:
    st.info("👋 Please upload `model.zip` to your GitHub repo to start.")



