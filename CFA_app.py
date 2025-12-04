import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import json
import os
import tempfile
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from hmmlearn.hmm import GaussianHMM
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="🏏 Cricket Form Analyzer - HMM",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .success-box {
        background-color: #d4edda;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if 'player_data' not in st.session_state:
    st.session_state.player_data = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

# ============================================================
# DATA EXTRACTION FUNCTION
# ============================================================
@st.cache_data(show_spinner=False)
def extract_player_data_from_json(uploaded_file, player_name, format_type='t20', max_files=1000):
    """Extract player match data from Cricsheet ZIP file."""
    player_matches = []

    try:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        with zipfile.ZipFile(tmp_path, 'r') as z:
            json_files = [f for f in z.namelist() if f.endswith('.json')]
            total_files = min(max_files, len(json_files))

            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, fname in enumerate(json_files[:max_files], 1):
                try:
                    with z.open(fname) as f:
                        content = f.read().decode('utf-8', errors='ignore')
                        if not content.strip():
                            continue

                        match_data = json.loads(content)
                        info = match_data.get("info", {})

                        # Check format
                        match_format = info.get("match_type", "").lower()
                        if format_type != "all" and match_format != format_type:
                            continue

                        # Check if player in match
                        player_in_match = False
                        if "players" in info:
                            for team, players in info["players"].items():
                                if player_name in players:
                                    player_in_match = True
                                    break

                        if not player_in_match:
                            continue

                        # Extract player innings
                        match_date = info.get("dates", [""])[0]

                        for innings in match_data.get("innings", []):
                            player_stats = {
                                'runs': 0,
                                'balls_faced': 0,
                                'fours': 0,
                                'sixes': 0,
                                'dots': 0,
                                'dismissed': False,
                                'date': match_date,
                                'format': match_format
                            }

                            found_in_innings = False

                            for over in innings.get("overs", []):
                                for delivery in over.get("deliveries", []):
                                    if delivery.get("batter", "") == player_name:
                                        found_in_innings = True
                                        runs = delivery.get("runs", {}).get("batter", 0)
                                        player_stats['runs'] += runs
                                        player_stats['balls_faced'] += 1

                                        if runs == 0:
                                            player_stats['dots'] += 1
                                        elif runs == 4:
                                            player_stats['fours'] += 1
                                        elif runs == 6:
                                            player_stats['sixes'] += 1

                                        if "wickets" in delivery:
                                            for wicket in delivery["wickets"]:
                                                if wicket.get("player_out") == player_name:
                                                    player_stats['dismissed'] = True

                            if found_in_innings and player_stats['balls_faced'] > 0:
                                player_matches.append(player_stats)

                    # Update progress
                    if idx % 50 == 0:
                        progress_bar.progress(idx / total_files)
                        status_text.text(f"Processing: {idx}/{total_files} files | Found {len(player_matches)} innings")

                except Exception as e:
                    continue

            progress_bar.progress(1.0)
            status_text.text(f"✅ Complete! Found {len(player_matches)} innings")

        # Cleanup temp file
        os.unlink(tmp_path)

        return player_matches

    except Exception as e:
        st.error(f"❌ Error: {e}")
        return []

# ============================================================
# FEATURE ENGINEERING FUNCTION
# ============================================================
def create_features(df):
    """Create ML features from raw data."""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Calculate strike rate
    df['strike_rate'] = (df['runs'] / df['balls_faced'] * 100).replace([np.inf, -np.inf], 0)

    # Boundary percentage
    df['boundary_pct'] = ((df['fours'] + df['sixes']) / df['balls_faced'] * 100).replace([np.inf, -np.inf], 0)

    # Dot ball percentage
    df['dot_pct'] = (df['dots'] / df['balls_faced'] * 100).replace([np.inf, -np.inf], 0)

    # Rolling averages (last 5 matches)
    df['rolling_avg_runs'] = df['runs'].rolling(window=5, min_periods=1).mean()
    df['rolling_avg_sr'] = df['strike_rate'].rolling(window=5, min_periods=1).mean()

    # Momentum (rate of change)
    df['momentum'] = df['runs'].diff().fillna(0)

    return df

# ============================================================
# HMM TRAINING FUNCTION
# ============================================================
def train_hmm_model(df, n_states=3, test_size=0.3):
    """Train HMM model with train/test split."""
    features = ['strike_rate', 'boundary_pct', 'dot_pct', 'rolling_avg_runs', 'rolling_avg_sr']

    X = df[features].values
    y = df['runs'].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    split_idx = int(len(df) * (1 - test_size))
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Train HMM
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=200,
        random_state=42
    )
    model.fit(X_train)

    # Get states
    train_states = model.predict(X_train)
    test_states = model.predict(X_test)

    # Calculate state means
    state_means = {}
    for state in range(n_states):
        state_runs = y_train[train_states == state]
        if len(state_runs) > 0:
            state_means[state] = np.mean(state_runs)
        else:
            state_means[state] = np.mean(y_train)

    # Predictions
    train_preds = np.array([state_means[s] for s in train_states])
    test_preds = np.array([state_means[s] for s in test_states])

    # Metrics
    results = {
        'model': model,
        'scaler': scaler,
        'state_means': state_means,
        'train': {
            'actual': y_train,
            'predicted': train_preds,
            'states': train_states,
            'mae': mean_absolute_error(y_train, train_preds),
            'rmse': np.sqrt(mean_squared_error(y_train, train_preds)),
            'r2': r2_score(y_train, train_preds) if len(y_train) > 1 else 0
        },
        'test': {
            'actual': y_test,
            'predicted': test_preds,
            'states': test_states,
            'mae': mean_absolute_error(y_test, test_preds),
            'rmse': np.sqrt(mean_squared_error(y_test, test_preds)),
            'r2': r2_score(y_test, test_preds) if len(y_test) > 1 else 0
        },
        'features': features
    }

    return results

# ============================================================
# MAIN APP
# ============================================================
def main():
    st.markdown('<h1 class="main-header">🏏 Cricket Form Analyzer with HMM</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # File upload
        st.subheader("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload Cricsheet ZIP file",
            type=['zip'],
            help="Download from cricsheet.org (t20s.zip, odis.zip, tests.zip)"
        )

        # Player settings
        st.subheader("👤 Player Settings")
        player_name = st.text_input(
            "Player Name",
            value="V Kohli",
            help="Enter exact name as in Cricsheet data"
        )

        format_type = st.selectbox(
            "Match Format",
            options=['t20', 'odi', 'test', 'all'],
            index=0
        )

        max_files = st.slider(
            "Max Files to Search",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100
        )

        # Model settings
        st.subheader("🤖 Model Settings")
        n_states = st.slider(
            "Number of HMM States",
            min_value=2,
            max_value=5,
            value=3,
            help="Form states (e.g., Poor, Average, Excellent)"
        )

        test_size = st.slider(
            "Test Set Size",
            min_value=0.1,
            max_value=0.4,
            value=0.3,
            step=0.05
        )

        extract_btn = st.button("🔍 Extract Data", type="primary", use_container_width=True)
        train_btn = st.button("🚀 Train Model", type="secondary", use_container_width=True)

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🎯 Model Training", "📈 Predictions", "🔮 Next Match"])

    # TAB 1: Data Overview
    with tab1:
        if extract_btn and uploaded_file is not None:
            with st.spinner(f"Extracting data for {player_name}..."):
                matches = extract_player_data_from_json(uploaded_file, player_name, format_type, max_files)

                if matches:
                    st.session_state.player_data = matches
                    df = pd.DataFrame(matches)
                    df = create_features(df)
                    st.session_state.df = df
                    st.success(f"✅ Extracted {len(matches)} innings for {player_name}")
                else:
                    st.warning("No data found. Check player name spelling.")

        if st.session_state.df is not None:
            df = st.session_state.df

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Innings", len(df))
            with col2:
                st.metric("Average Runs", f"{df['runs'].mean():.1f}")
            with col3:
                st.metric("Highest Score", df['runs'].max())
            with col4:
                st.metric("Avg Strike Rate", f"{df['strike_rate'].mean():.1f}")

            st.subheader("📋 Match Data")
            st.dataframe(
                df[['date', 'runs', 'balls_faced', 'fours', 'sixes', 'strike_rate', 'dismissed']].tail(20),
                use_container_width=True
            )

            # Runs distribution chart
            st.subheader("📊 Runs Distribution")
            fig = px.histogram(df, x='runs', nbins=20, title="Distribution of Runs Scored")
            fig.update_layout(xaxis_title="Runs", yaxis_title="Frequency")
            st.plotly_chart(fig, use_container_width=True)

            # Runs over time
            st.subheader("📈 Performance Timeline")
            fig2 = px.line(df, x='date', y='runs', title="Runs Over Time", markers=True)
            fig2.add_scatter(x=df['date'], y=df['rolling_avg_runs'], name="5-Match Avg", line=dict(dash='dash'))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("👈 Upload a ZIP file and extract data to begin analysis")

    # TAB 2: Model Training
    with tab2:
        if train_btn and st.session_state.df is not None:
            df = st.session_state.df

            if len(df) < 10:
                st.error("Need at least 10 innings to train model")
            else:
                with st.spinner("Training HMM model..."):
                    results = train_hmm_model(df, n_states=n_states, test_size=test_size)
                    st.session_state.predictions = results
                    st.success("✅ Model trained successfully!")

        if st.session_state.predictions is not None:
            results = st.session_state.predictions

            st.subheader("📊 Model Performance")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Training Set**")
                st.metric("MAE", f"{results['train']['mae']:.2f}")
                st.metric("RMSE", f"{results['train']['rmse']:.2f}")
                st.metric("R² Score", f"{results['train']['r2']:.3f}")

            with col2:
                st.markdown("**Test Set**")
                st.metric("MAE", f"{results['test']['mae']:.2f}")
                st.metric("RMSE", f"{results['test']['rmse']:.2f}")
                st.metric("R² Score", f"{results['test']['r2']:.3f}")

            # State analysis
            st.subheader("🎯 Form States Analysis")
            state_names = {0: "Poor Form", 1: "Average Form", 2: "Excellent Form"}
            if n_states > 3:
                state_names = {i: f"State {i+1}" for i in range(n_states)}

            state_df = pd.DataFrame({
                'State': [state_names.get(k, f"State {k}") for k in results['state_means'].keys()],
                'Expected Runs': list(results['state_means'].values())
            })
            st.dataframe(state_df, use_container_width=True)

            # Transition matrix
            st.subheader("🔄 State Transition Probabilities")
            trans_matrix = results['model'].transmat_
            trans_df = pd.DataFrame(
                trans_matrix,
                columns=[state_names.get(i, f"State {i}") for i in range(n_states)],
                index=[state_names.get(i, f"State {i}") for i in range(n_states)]
            )
            st.dataframe(trans_df.style.format("{:.2%}"), use_container_width=True)
        else:
            st.info("👈 Train the model to see performance metrics")

    # TAB 3: Predictions Visualization
    with tab3:
        if st.session_state.predictions is not None:
            results = st.session_state.predictions

            st.subheader("📈 Actual vs Predicted Runs")

            # Create comparison dataframe
            train_df = pd.DataFrame({
                'Index': range(len(results['train']['actual'])),
                'Actual': results['train']['actual'],
                'Predicted': results['train']['predicted'],
                'Set': 'Training'
            })

            test_df = pd.DataFrame({
                'Index': range(len(results['train']['actual']), len(results['train']['actual']) + len(results['test']['actual'])),
                'Actual': results['test']['actual'],
                'Predicted': results['test']['predicted'],
                'Set': 'Testing'
            })

            comparison_df = pd.concat([train_df, test_df])

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=comparison_df['Index'], y=comparison_df['Actual'], 
                                      mode='lines+markers', name='Actual', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=comparison_df['Index'], y=comparison_df['Predicted'], 
                                      mode='lines+markers', name='Predicted', line=dict(color='red', dash='dash')))

            # Add vertical line separating train/test
            fig.add_vline(x=len(train_df)-0.5, line_dash="dot", annotation_text="Train|Test Split")
            fig.update_layout(title="Actual vs Predicted Runs", xaxis_title="Match Index", yaxis_title="Runs")
            st.plotly_chart(fig, use_container_width=True)

            # Residuals plot
            st.subheader("📉 Prediction Errors (Test Set)")
            residuals = results['test']['actual'] - results['test']['predicted']
            fig_res = px.bar(x=range(len(residuals)), y=residuals, 
                            title="Prediction Residuals", labels={'x': 'Match', 'y': 'Error (Actual - Predicted)'})
            fig_res.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_res, use_container_width=True)
        else:
            st.info("Train the model first to see predictions")

    # TAB 4: Next Match Prediction
    with tab4:
        if st.session_state.predictions is not None and st.session_state.df is not None:
            results = st.session_state.predictions
            df = st.session_state.df

            st.subheader("🔮 Next Match Prediction")

            # Get current state from last match
            features = results['features']
            last_features = df[features].iloc[-1:].values
            last_scaled = results['scaler'].transform(last_features)
            current_state = results['model'].predict(last_scaled)[0]

            state_names = {0: "Poor Form", 1: "Average Form", 2: "Excellent Form"}
            current_state_name = state_names.get(current_state, f"State {current_state}")
            predicted_runs = results['state_means'][current_state]

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"""
                <div class="success-box">
                <h3>Current Form State</h3>
                <h2>{current_state_name}</h2>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="success-box">
                <h3>Predicted Runs (Next Match)</h3>
                <h2>{predicted_runs:.1f}</h2>
                </div>
                """, unsafe_allow_html=True)

            # Confidence intervals
            st.subheader("📊 Prediction Confidence")
            all_runs = df['runs'].values
            std_dev = np.std(all_runs)

            ci_50_low = max(0, predicted_runs - 0.675 * std_dev)
            ci_50_high = predicted_runs + 0.675 * std_dev
            ci_90_low = max(0, predicted_runs - 1.645 * std_dev)
            ci_90_high = predicted_runs + 1.645 * std_dev

            st.markdown(f"""
            - **50% Confidence Interval**: {ci_50_low:.0f} - {ci_50_high:.0f} runs
            - **90% Confidence Interval**: {ci_90_low:.0f} - {ci_90_high:.0f} runs
            """)

            # State transition probabilities
            st.subheader("📈 Form Transition Outlook")
            trans_probs = results['model'].transmat_[current_state]

            outlook_df = pd.DataFrame({
                'Next State': [state_names.get(i, f"State {i}") for i in range(len(trans_probs))],
                'Probability': trans_probs,
                'Expected Runs': [results['state_means'][i] for i in range(len(trans_probs))]
            })

            fig = px.bar(outlook_df, x='Next State', y='Probability', 
                        title="Probability of Next Match Form State",
                        color='Expected Runs', color_continuous_scale='RdYlGn')
            fig.update_layout(yaxis_tickformat='.0%')
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(outlook_df.style.format({'Probability': '{:.1%}', 'Expected Runs': '{:.1f}'}), 
                        use_container_width=True)
        else:
            st.info("Train the model first to see next match predictions")

if __name__ == "__main__":
    main()
