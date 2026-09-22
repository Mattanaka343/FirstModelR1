import streamlit as st
import numpy as np
import pandas as pd
import os
import pickle
import re
from glob import glob
from pathlib import Path

DEFAULT_N_FFT = 10
SENSOR_NAMES_1 = ["pitch1", "yaw1", "roll1", "pitch2", "yaw2", "roll2"]
SENSOR_NAMES_2 = ["f1", "f2", "f3", "f4", "f5", "pitch3"]
SENSOR_NAMES_COMBINED = SENSOR_NAMES_1 + SENSOR_NAMES_2


# ====================================================================
# PREPROCESSING UTILITIES
# ====================================================================

def extract_features(data, sensor_names, n_fft=DEFAULT_N_FFT):
    """
    Extract statistical and FFT features from sensor data.
    
    Parameters:
    - data: numpy array of shape (n_samples, n_timesteps, n_sensors)
    - sensor_names: list of sensor names
    - n_fft: number of FFT components to keep
    
    Returns:
    - DataFrame with extracted features
    """
    features = {}

    for i, sensor in enumerate(sensor_names):
        x = data[:, :, i]

        # Mean
        features[f"{sensor}_mean"] = np.mean(x, axis=1)

        # Standard deviation
        features[f"{sensor}_std"] = np.std(x, axis=1)

        # FFT
        fft = np.fft.rfft(x, axis=1)

        # Magnitude and remove DC component
        fft_magnitude = np.abs(fft)[:, 1:]

        # Keep only first n_fft components
        fft_magnitude = fft_magnitude[:, :n_fft]

        for j in range(n_fft):
            features[f"{sensor}_fft_{j+1}"] = fft_magnitude[:, j]

    return pd.DataFrame(features)


def preprocess_observation(data, sensor_config_names, n_fft=DEFAULT_N_FFT):
    """
    Preprocess a single observation using the defined sensor configuration.
    
    Parameters:
    - data: numpy array of shape (n_timesteps, n_features) from one .npy file
    - sensor_config_names: list of sensor names for this configuration
    - n_fft: number of FFT components
    
    Returns:
    - DataFrame with preprocessed features
    """
    data_reshaped = data[np.newaxis, :, :]
    features_df = extract_features(data_reshaped, sensor_config_names, n_fft=n_fft)
    return features_df


def align_features_with_training(features_df, expected_columns):
    """
    Align the extracted features with the expected training columns.
    Fill missing columns with 0.0 and drop extra columns.
    """
    features_df = features_df.reindex(columns=expected_columns, fill_value=0.0)
    return features_df


def build_expected_feature_columns(sensor_names, n_fft=DEFAULT_N_FFT):
    feature_cols = []
    for sensor in sensor_names:
        feature_cols.append(f"{sensor}_mean")
        feature_cols.append(f"{sensor}_std")
        for j in range(n_fft):
            feature_cols.append(f"{sensor}_fft_{j+1}")
    return feature_cols


# ====================================================================
# DATA LOADING UTILITIES
# ====================================================================

def get_available_activities():
    """
    Scan the raw data directory to find all available activities.
    Returns a dictionary with activity IDs and their file paths.
    """
    data_dir = Path("Data/d01_raw_data")
    activities = {}
    
    if data_dir.exists():
        files = sorted(glob(str(data_dir / "*.npy")))
        for file in files:
            filename = os.path.basename(file)
            match = re.match(r"(\d+)_([12]).npy", filename)
            if match:
                activity_id = match.group(1)
                sensor_config = match.group(2)
                if activity_id not in activities:
                    activities[activity_id] = {}
                activities[activity_id][sensor_config] = file
    
    return activities


def load_raw_data(file_path):
    """Load a .npy file."""
    return np.load(file_path)


def simulate_observation(activity_id, activities_dict):
    """
    Create a new synthetic observation for the selected activity.

    This is NOT just a copy of an existing record.
    We take a random real sample from the activity, then apply controlled
    perturbations (noise, trend, phase-shifted sinusoidal modulation) to
    generate a new but realistic signal that still reflects the activity's
    sensor behavior.
    """
    if activity_id not in activities_dict:
        return None, None

    activity_files = activities_dict[activity_id]
    data1 = load_raw_data(activity_files['1']) if '1' in activity_files else None
    data2 = load_raw_data(activity_files['2']) if '2' in activity_files else None

    if data1 is None and data2 is None:
        return None, None

    rng = np.random.default_rng()

    def synthesize_group(raw_group):
        if raw_group is None:
            return None

        base_idx = int(rng.integers(0, raw_group.shape[0]))
        base_signal = raw_group[base_idx].copy()
        t = np.arange(base_signal.shape[0])

        synthetic = np.zeros_like(base_signal, dtype=float)
        for sensor_idx in range(base_signal.shape[1]):
            channel = base_signal[:, sensor_idx].astype(float)
            channel_std = float(np.std(channel))
            if channel_std == 0:
                channel_std = 1.0

            noise = rng.normal(0, 0.08 * channel_std, size=channel.shape[0])
            modulation = 0.12 * channel_std * np.sin(
                2 * np.pi * rng.uniform(0.5, 3.0) * t / max(len(t), 1) + rng.uniform(0, 2 * np.pi)
            )
            drift = 0.04 * channel_std * np.linspace(-1, 1, len(channel))
            synthetic[:, sensor_idx] = channel + noise + modulation + drift

        return synthetic

    synthetic1 = synthesize_group(data1)
    synthetic2 = synthesize_group(data2)

    if synthetic1 is not None and synthetic2 is not None:
        combined_data = np.concatenate([synthetic1, synthetic2], axis=1)
    elif synthetic1 is not None:
        missing = np.zeros((synthetic1.shape[0], len(SENSOR_NAMES_2)))
        combined_data = np.concatenate([synthetic1, missing], axis=1)
    else:
        missing = np.zeros((synthetic2.shape[0], len(SENSOR_NAMES_1)))
        combined_data = np.concatenate([missing, synthetic2], axis=1)

    return combined_data, "synthetic"


# ====================================================================
# MODEL LOADING UTILITIES
# ====================================================================

def get_available_models():
    """
    Scan the Models directory to find all available trained models.
    Returns a list of model filenames.
    """
    models_dir = Path("Models")
    models = []
    
    if models_dir.exists():
        models = sorted([f.name for f in models_dir.glob("*.pkl")])
    
    return models


def load_model(model_name):
    """
    Load a trained model from the Models directory.
    
    Parameters:
    - model_name: filename of the model
    
    Returns:
    - Loaded model object
    """
    model_path = Path("Models") / model_name
    
    if not model_path.exists():
        return None
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def make_prediction(model, features_df):
    """
    Make a prediction using the trained model.
    
    Parameters:
    - model: trained model object (should have predict method)
    - features_df: preprocessed features DataFrame
    
    Returns:
    - Prediction (class or probability)
    """
    try:
        if isinstance(features_df, pd.DataFrame):
            X = features_df.values
        else:
            X = features_df
        
        prediction = model.predict(X)
        
        if isinstance(prediction, np.ndarray):
            if prediction.ndim > 1:
                prediction = prediction[0]
            else:
                prediction = prediction[0] if len(prediction) > 0 else prediction
        
        return prediction
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None


def normalize_activity_label(value):
    """Normalize activity labels to a comparable format.

    Models may return integers like 0, 5, 15, while the dataset labels are
    stored as strings with zero padding such as 000, 005, 015.
    """
    if value is None:
        return "0"
    value_str = str(value).strip()
    if value_str == "":
        return "0"
    value_str = value_str.lstrip("0") or "0"
    return value_str


# ====================================================================
# STREAMLIT APP
# ====================================================================

def main():
    st.set_page_config(page_title="Activity Classification", layout="wide")
    st.title("Activity Classification Interface")
    
    st.markdown("""
    This application allows you to:
    1. **Simulate** a brand-new observation for the selected activity
    2. **Preprocess** the data using FFT and statistical features
    3. **Predict** the activity class using a trained model
    """)
    
    st.sidebar.header("Configuration")
    activities_dict = get_available_activities()
    available_models = get_available_models()

    activity_options = sorted(list(activities_dict.keys()))
    if not activity_options:
        st.error("No activity data found in Data/d01_raw_data/")
        return

    selected_activity = st.sidebar.selectbox(
        "Select Activity",
        activity_options,
        help="Choose the activity whose signal you want to simulate"
    )

    if not available_models:
        st.warning("No trained models found in Models/ directory. Predictions will not be available.")
        selected_model = None
    else:
        selected_model = st.sidebar.selectbox(
            "Select Model",
            available_models,
            help="Choose a trained model for making predictions"
        )

    st.sidebar.info(
        "Simulation is generated synthetically: the app starts from a real activity sample and adds small controlled perturbations so the result is a new but realistic observation."
    )
    st.sidebar.info("FFT is fixed at 10 components because the trained models were built with that feature size.")
    n_fft = DEFAULT_N_FFT

    col1, col2 = st.columns(2)
    with col1:
        st.header("Step 1: Simulate Observation")
        if st.button("🎲 Simulate Random Observation", key="simulate"):
            st.session_state.simulated = True
            st.session_state.combined_data, st.session_state.obs_idx = simulate_observation(
                selected_activity, activities_dict
            )
    with col2:
        st.header("Step 2: Preprocess Data")

    if "simulated" in st.session_state and st.session_state.simulated:
        st.subheader("📊 Simulated Observation Details")
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Activity ID:** {selected_activity}")
            st.info(f"**Observation Index:** {st.session_state.obs_idx}")
        with col2:
            if st.session_state.combined_data is not None:
                st.success(f"**Data Shape:** {st.session_state.combined_data.shape}")
                st.success(f"**Timesteps:** {st.session_state.combined_data.shape[0]}")

        if st.session_state.combined_data is not None:
            signal_df = pd.DataFrame(st.session_state.combined_data, columns=SENSOR_NAMES_COMBINED)

            st.subheader("📈 Simulated Signal Plot")
            st.caption("This is the sampled signal that was randomly selected from the activity recordings.")
            st.line_chart(signal_df.iloc[:, :6])
            st.line_chart(signal_df.iloc[:, 6:])

        if st.checkbox("Show Raw Data Preview"):
            if st.session_state.combined_data is not None:
                st.write("First 5 timesteps of sensor data:")
                st.dataframe(
                    pd.DataFrame(
                        st.session_state.combined_data[:5],
                        columns=SENSOR_NAMES_COMBINED
                    )
                )

        st.subheader("🔧 Preprocessing")
        
        if st.button("Preprocess Data"):
            with st.spinner("Preprocessing data..."):
                features_df = preprocess_observation(
                    st.session_state.combined_data,
                    SENSOR_NAMES_COMBINED,
                    n_fft=n_fft
                )
                feature_cols = build_expected_feature_columns(SENSOR_NAMES_COMBINED, n_fft=n_fft)
                features_df = align_features_with_training(features_df, feature_cols)
                st.session_state.preprocessed_features = features_df
                st.session_state.features_ready = True

        if "features_ready" in st.session_state and st.session_state.features_ready:
            st.success("✅ Data preprocessed successfully!")
            st.write(f"**Total Features Extracted:** {len(st.session_state.preprocessed_features.columns)}")
            
            if st.checkbox("Show Extracted Features"):
                st.dataframe(st.session_state.preprocessed_features, use_container_width=True)

    st.header("Step 3: Make Prediction")
    
    if "features_ready" in st.session_state and st.session_state.features_ready:
        if selected_model is None:
            st.warning("No model available. Please add a trained model to the Models/ directory.")
        else:
            if st.button("🎯 Make Prediction"):
                with st.spinner(f"Loading model: {selected_model}..."):
                    model = load_model(selected_model)
                    if model is not None:
                        with st.spinner("Making prediction..."):
                            prediction = make_prediction(model, st.session_state.preprocessed_features)
                            if prediction is not None:
                                st.session_state.prediction = prediction
                                st.session_state.prediction_made = True
        
        if "prediction_made" in st.session_state and st.session_state.prediction_made:
            st.subheader("🎯 Prediction Results")
            pred_label = normalize_activity_label(st.session_state.prediction)
            gt_label = normalize_activity_label(selected_activity)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Activity", pred_label)
            with col2:
                st.metric("Actual Activity (Ground Truth)", gt_label)
            with col3:
                match = pred_label == gt_label
                status = "✅ Correct" if match else "❌ Incorrect"
                st.metric("Match", status)
            st.info(f"Prediction made using model: **{selected_model}**")
    else:
        st.info("👈 Please simulate and preprocess data first to make predictions.")

    st.divider()
    st.markdown("""
    ---
    **How this works:**
    1. A real observation is sampled from the selected activity's raw recordings as a starting template.
    2. A brand-new synthetic signal is generated by adding controlled noise, drift, and small sinusoidal modulation to the template.
    3. The selected signal is plotted so you can inspect the simulated motion pattern.
    4. The raw signal is then transformed into summary features: mean, standard deviation, and the first 10 FFT magnitudes for each sensor.
    5. Those features are passed to the model chosen from the `Models/` folder.
    """)


if __name__ == "__main__":
    main()
