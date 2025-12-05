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

# ---------- GitHub ZIP URLs (update to your actual paths) ----------
# DATA zip in your GitHub repo
DATA_ZIP_URL = (
    "https://github.com/aksmamg014/Cricket_Form_Analyzer/"
    "blob/main/t20s.zip?raw=true"
)

# MODEL zip in your GitHub repo (put your real path here)
MODEL_ZIP_URL = (
    "https://github.com/aksmamg014/Cricket_Form_Analyzer/"
    "blob/main/models.zip?raw=true"
)

# Where to extract
MODEL_DIR = "models/rf_cricket_score"
DATA_DIR = "t20s_data"

# ==========================================
# CACHED SETUP FUNCTIONS
# ==========================================

@st.cache_resource
def download_and_extract_zip(url: str, target_dir: str) -> bool:
    """Download a zip from GitHub and extract into target_dir."""
    try:
        # If already extracted with files, skip download
        if os.path.exists(target_dir) and len(os.listdir(target_dir)) > 0:
            return True

        resp = requests.get(url, timeout=30)
        resp.raise_for_status()

        z = zipfile.ZipFile(io.BytesIO(resp.content))
        # For data, extract into target_dir; for model we often want original paths
        z.extractall(target_dir)
        return True
    except Exception as e:
        st.error(f"❌ Failed to download/extract from {url}: {e}")
        return False

@st.cache_resource
def setup_model_files() -> bool:
    """
    Ensure model files are present by downloading models.zip from GitHub
    and extracting it so that MODEL_DIR exists and has .joblib files.
    """
    # If already there, done
    if os.path.exists(MODEL_DIR) and len(os.listdir(MODEL_DIR)) > 0:
        return True

    # Download & extract into a temp dir, then check MODEL_DIR
    ok = download_and_extract_zip(MODEL_ZIP_URL, ".")  # keep original paths inside zip
    if not ok:
        return False

    if os.path.exists(MODEL_DIR) and len(os.listdir(MODEL_DIR)) > 0:
        return True

    st.error(f"Unzipped model from GitHub but `{MODEL_DIR}` is missing or empty. "
             "Check the folder structure inside models.zip.")
    return False

@st.cache_resource
def download_data_on_startup() -> bool:
    """
    Automatically download & extract t20s.zip from GitHub into DATA_DIR.
    """
    return download_and_extract_zip(DATA_ZIP_URL, DATA_DIR)

@st.cache_resource
def load_model_artifacts(model_dir):
    """Loads model, feature_names, metadata from downloaded model files."""
    if not setup_model_files():
        return None, None, {}

    if not os.path.exists(model_dir):
        return None, None, {}

    try:
        files = [f for f in os.listdir(model_dir)
                 if f.startswith("rf_model_") and f.endswith(".joblib")]
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

        if isinstance(payload, dict) and "model" in payload:
            model_obj = payload["model"]
            features = payload.get("feature_names")
            meta = payload.get("metadata", {})
        else:
            model_obj = payload

        # Normalize feature_names to a plain Python list
        if features is not None:
            if hasattr(features, "tolist"):
                features = features.tolist()
            elif hasattr(features, "columns"):
                features = features.columns.tolist()

        return model_obj, features, meta
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, {}

@st.cache_data
def load_cricket_data():
    """Read player-level CSV from DATA_DIR (downloaded from GitHub zip)."""
    if not os.path.exists(DATA_DIR):
        return None

    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    if not csv_files:
        st.error(f"No CSV files found in {DATA_DIR}")
        return None

    try:
        path = os.path.join(DATA_DIR, csv_files[0])
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None
