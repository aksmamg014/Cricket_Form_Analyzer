import streamlit as st
import pandas as pd
import joblib
import os
import requests
import zipfile
import io
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================
st.set_page_config(page_title="Cricket Score Predictor", page_icon="🏏", layout="wide")

# Placeholder for your GitHub Raw URL (User must update this)
# Example: "https://github.com/yourusername/your-repo/raw/main/data/cricket_data.zip"
GITHUB_ZIP_URL = "https://github.com/YOUR_USER/YOUR_REPO/raw/main/data/data.zip"

# Path to the model directory created in your previous step
MODEL_DIR = "models/rf_cricket_score"

# ==========================================
# CACHED FUNCTIONS
# ==========================================

@st.cache_resource
def load_latest_model(model_dir):
    """Loads the latest .joblib model from the directory."""
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
        # Handle both dictionary format (from your code) and raw model
        if isinstance(payload, dict) and 'model' in payload:
            return payload['model'], payload.get('feature_names'), payload.get('metadata')
        else:
            return payload, None, {} # Fallback if raw model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

@st.cache_resource
def load_feature_names(model_dir):
    """Loads feature names separately if they exist."""
    path = os.path.join(model_dir, "feature_names.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    return None

@st.cache_data
def download_and_extract_github_data(url):
    """Downloads zip from GitHub and extracts to a temp folder."""
    if "YOUR_USER" in url: # Check if placeholder is still there
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
    
    # Fallback to separate feature file if not in model dict
    if not meta_features:
        meta_features = load_feature_names(MODEL_DIR)
    
    if model:
        st.success(f"✅ Model Loaded!")
        if metadata and 'timestamp' in metadata:
            st.caption(f"Trained: {metadata['timestamp']}")
            st.metric("CV MAE", f"{metadata.get('cv_mae_mean', 0):.2f}")
    else:
        st.error("❌ No model found in directory.")
    
    st.markdown("---")
    st.subheader("🔗 Data Source")
    github_url = st.text_input("GitHub Zip URL", value=GITHUB_ZIP_URL)
    
    if st.button("Download Data"):
        data_path = download_and_extract_github_data(github_url)
        if data_path:
            st.success(f"Data extracted to `{data_path}`")
        else:
            st.warning("Please provide a valid GitHub 'Raw' URL.")

# --- Main Prediction Area ---

if model and meta_features:
    st.subheader("🔮 Make a Prediction")
    
    # Create a form for inputs
    with st.form("prediction_form"):
        st.markdown("### Input Features")
        
        # Dynamically generate input fields based on feature names
        # We arrange them in columns for better layout
        cols = st.columns(3)
        input_data = {}
        
        for i, feature in enumerate(meta_features):
            with cols[i % 3]:
                # Heuristic to guess input type based on name
                if 'runs' in feature.lower() or 'score' in feature.lower():
                    input_data[feature] = st.number_input(feature, min_value=0, value=100)
                elif 'wickets' in feature.lower():
                    input_data[feature] = st.number_input(feature, min_value=0, max_value=10, value=2)
                elif 'overs' in feature.lower():
                    input_data[feature] = st.number_input(feature, min_value=0.0, max_value=50.0, value=10.0)
                else:
                    # Default to number input for unknown features (safe for ML models)
                    input_data[feature] = st.number_input(feature, value=0.0)

        submit = st.form_submit_button("🚀 Predict Score")

    if submit:
        # Convert inputs to DataFrame
        df_input = pd.DataFrame([input_data])
        
        # Predict
        prediction = model.predict(df_input)[0]
        
        st.markdown("---")
        st.metric(label="Predicted Score", value=f"{prediction:.0f} Runs")
        
else:
    st.info("👋 Please ensure your model is saved in `models/rf_cricket_score` to continue.")
