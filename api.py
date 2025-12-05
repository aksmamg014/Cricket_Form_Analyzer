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

# URLs and paths
DATA_ZIP_URL = "https://github.com/aksmamg014/Cricket_Form_Analyzer/blob/main/t20s.zip?raw=true"
MODEL_ZIP_FILE = "models.zip" 
MODEL_DIR = "models/rf_cricket_score"
DATA_DIR = "t20s_data"

# ==========================================
# CACHED SETUP FUNCTIONS
# ==========================================

@st.cache_resource
def setup_model_files():
    """Ensures model files are present by unzipping models.zip"""
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
    """Downloads and extracts t20s.zip from GitHub"""
    if os.path.exists(DATA_DIR) and len(os.listdir(DATA_DIR)) > 0:
        return True

    try:
        response = requests.get(DATA_ZIP_URL, timeout=30)
        response.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(response.content))
        z.extractall(DATA_DIR)
        return True
    except Exception as e:
        st.error(f"❌ Data download failed: {e}")
        return False

@st.cache_resource
def load_model_artifacts(model_dir):
    """Loads model, features, and metadata"""
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
    """Loads feature names from separate file if needed"""
    path = os.path.join(model_dir, "feature_names.joblib")
    if os.path.exists(path):
        names = joblib.load(path)
        if hasattr(names, 'tolist'):
            return names.tolist()
        return names
    return None

@st.cache_data
def load_cricket_data():
    """Loads the cricket CSV data from extracted folder"""
    if not os.path.exists(DATA_DIR):
        return None
    
    # Find CSV files in the data directory
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    
    if not csv_files:
        st.error(f"No CSV files found in {DATA_DIR}")
        return None
    
    # Load the first CSV (adjust if you have multiple files)
    try:
        data_path = os.path.join(DATA_DIR, csv_files[0])
        df = pd.read_csv(data_path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def prepare_player_features(player_df, feature_names):
    """
    Prepares features for prediction based on player's recent performance.
    This function calculates aggregated stats from the player's history.
    Adjust this logic based on your actual feature engineering.
    """
    features = {}
    
    # Example feature calculations (customize based on your model's features)
    for feature in feature_names:
        feature_lower = feature.lower()
        
        # Calculate based on feature name patterns
        if 'avg' in feature_lower or 'mean' in feature_lower:
            if 'runs' in feature_lower or 'score' in feature_lower:
                features[feature] = player_df['runs'].mean() if 'runs' in player_df.columns else 0
            elif 'balls' in feature_lower:
                features[feature] = player_df['balls_faced'].mean() if 'balls_faced' in player_df.columns else 0
        elif 'last' in feature_lower:
            if 'runs' in feature_lower:
                features[feature] = player_df['runs'].iloc[-1] if len(player_df) > 0 and 'runs' in player_df.columns else 0
        elif 'total' in feature_lower:
            if 'runs' in feature_lower:
                features[feature] = player_df['runs'].sum() if 'runs' in player_df.columns else 0
        else:
            # Default: use mean of numeric columns or 0
            features[feature] = 0
    
    return pd.DataFrame([features])

# ==========================================
# MAIN APP
# ==========================================

st.title("🏏 Cricket Player Score Predictor")
st.markdown("Select a player and predict their next match performance")

# --- Auto-Run Setup ---
data_ready = download_data_on_startup()
model, meta_features, metadata = load_model_artifacts(MODEL_DIR)

if (meta_features is None or len(meta_features) == 0) and model is not None:
    meta_features = load_feature_names_fallback(MODEL_DIR)

cricket_data = load_cricket_data() if data_ready else None

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ System Status")
    
    if data_ready:
        st.success("✅ Data Loaded")
    else:
        st.error("❌ Data Missing")
        
    if model is not None:
        st.success("✅ Model Loaded")
        if metadata.get('timestamp'):
            st.caption(f"Version: {metadata.get('timestamp')}")
            st.metric("Model MAE", f"{metadata.get('cv_mae_mean', 0):.2f}")
    else:
        st.error("❌ Model Missing")
    
    if cricket_data is not None:
        st.info(f"📊 {len(cricket_data)} records available")

# --- Main Prediction Interface ---
if model is not None and meta_features is not None and cricket_data is not None:
    
    # Identify player column (adjust column name based on your CSV)
    player_column = None
    possible_names = ['player', 'player_name', 'batsman', 'striker', 'batter']
    for col in cricket_data.columns:
        if col.lower() in possible_names:
            player_column = col
            break
    
    if player_column is None:
        st.error("❌ Could not find player name column in data. Available columns:")
        st.write(cricket_data.columns.tolist())
    else:
        # Get unique player names
        players = sorted(cricket_data[player_column].dropna().unique())
        
        st.subheader("🎯 Select Player")
        
        # Player selection
        selected_player = st.selectbox(
            "Choose a player:",
            options=players,
            index=0 if len(players) > 0 else None
        )
        
        if selected_player:
            # Filter player data
            player_data = cricket_data[cricket_data[player_column] == selected_player]
            
            # Display player stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Matches", len(player_data))
            with col2:
                if 'runs' in cricket_data.columns:
                    st.metric("Average Runs", f"{player_data['runs'].mean():.1f}")
            with col3:
                if 'runs' in cricket_data.columns:
                    st.metric("Total Runs", f"{player_data['runs'].sum():.0f}")
            
            # Predict button
            if st.button("🚀 Predict Next Match Score", type="primary"):
                try:
                    # Prepare features from player history
                    input_features = prepare_player_features(player_data, meta_features)
                    
                    # Make prediction
                    prediction = model.predict(input_features)[0]
                    
                    # Display result
                    st.markdown("---")
                    st.markdown(f"### 🎯 Predicted Score for {selected_player}")
                    st.markdown(f"## **{int(prediction)} Runs**")
                    
                    # Show recent performance
                    with st.expander("📈 Recent Performance"):
                        if 'runs' in player_data.columns:
                            recent = player_data.tail(5)[['runs'] if 'runs' in player_data.columns else player_data.columns[:3]]
                            st.dataframe(recent)
                        
                except Exception as e:
                    st.error(f"Prediction Error: {e}")
                    st.write("Debug: Feature mismatch. Check your feature engineering.")

else:
    st.warning("⚠️ System initializing... Please wait.")
    if model is None:
        st.error("Model not loaded. Check that models.zip exists in repo.")
    if cricket_data is None:
        st.error("Data not loaded. Check t20s.zip download.")
