"""
Preprocessing utilities for sensor data.

This module provides reusable preprocessing functions for activity classification.
It can be imported in both the Streamlit app and training scripts.
"""

import numpy as np
import pandas as pd
from pathlib import Path


# Sensor configuration constants
SENSOR_CONFIG_1 = ["pitch1", "yaw1", "roll1", "pitch2", "yaw2", "roll2"]
SENSOR_CONFIG_2 = ["f1", "f2", "f3", "f4", "f5", "pitch3"]
SENSOR_CONFIG_COMBINED = SENSOR_CONFIG_1 + SENSOR_CONFIG_2

DEFAULT_N_FFT = 10


def extract_features(data, sensor_names, n_fft=DEFAULT_N_FFT):
    """
    Extract statistical and FFT features from sensor time-series data.
    
    This function computes mean, standard deviation, and FFT frequency components
    for each sensor channel. It follows the preprocessing approach from FirstModel.qmd.
    
    Parameters
    ----------
    data : np.ndarray
        Input data of shape (n_samples, n_timesteps, n_sensors).
        Each sample is a time-series of sensor readings.
    
    sensor_names : list of str
        Names of the sensors in the same order as the last dimension of data.
        Example: ['pitch1', 'yaw1', 'roll1', 'pitch2', 'yaw2', 'roll2', 'f1', ...]
    
    n_fft : int
        Number of FFT frequency components to extract per sensor (default: 10).
        Higher values capture more frequency information but increase dimensionality.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with extracted features. Columns follow the pattern:
        {sensor}_mean, {sensor}_std, {sensor}_fft_1, ..., {sensor}_fft_n
        
        With n_fft=10 and m sensors, output has shape (n_samples, m * 12).
    
    Example
    -------
    >>> # Load raw IMU data (e.g., from FirstModel.qmd)
    >>> data = np.load('Data/d01_raw_data/000_1.npy')  # shape: (n_obs, n_timesteps, 6)
    >>> sensor_names = ['pitch1', 'yaw1', 'roll1', 'pitch2', 'yaw2', 'roll2']
    >>> features = extract_features(data, sensor_names, n_fft=10)
    >>> features.shape
    (500, 72)  # 500 samples, 72 features (6 sensors * 12 features each)
    """
    features = {}

    for i, sensor in enumerate(sensor_names):
        x = data[:, :, i]

        # Mean of the signal
        features[f"{sensor}_mean"] = np.mean(x, axis=1)

        # Standard deviation of the signal
        features[f"{sensor}_std"] = np.std(x, axis=1)

        # FFT frequency components
        fft = np.fft.rfft(x, axis=1)
        fft_magnitude = np.abs(fft)[:, 1:]  # Remove DC component

        # Keep only first n_fft components
        fft_magnitude = fft_magnitude[:, :n_fft]

        for j in range(n_fft):
            features[f"{sensor}_fft_{j+1}"] = fft_magnitude[:, j]

    return pd.DataFrame(features)


def build_expected_columns(sensor_names=None, n_fft=DEFAULT_N_FFT):
    """
    Build the expected column names for preprocessed features.
    
    This is useful for ensuring feature alignment across different datasets.
    
    Parameters
    ----------
    sensor_names : list of str, optional
        Sensor names. If None, uses the combined sensor configuration.
    
    n_fft : int
        Number of FFT components (default: 10).
    
    Returns
    -------
    list of str
        Ordered list of feature column names.
    """
    if sensor_names is None:
        sensor_names = SENSOR_CONFIG_COMBINED
    
    columns = []
    for sensor in sensor_names:
        columns.append(f"{sensor}_mean")
        columns.append(f"{sensor}_std")
        for j in range(n_fft):
            columns.append(f"{sensor}_fft_{j+1}")
    
    return columns


def align_features_with_training(features_df, expected_columns=None, n_fft=DEFAULT_N_FFT):
    """
    Align extracted features with the expected training data format.
    
    Ensures that:
    - All expected columns are present (missing ones filled with 0.0)
    - Extra columns are removed
    - Column order matches the training data
    
    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame with extracted features.
    
    expected_columns : list of str, optional
        Expected column names. If None, uses default combined sensor configuration.
    
    n_fft : int
        Number of FFT components (default: 10).
    
    Returns
    -------
    pd.DataFrame
        Aligned feature DataFrame ready for model input.
    """
    if expected_columns is None:
        expected_columns = build_expected_columns(n_fft=n_fft)
    
    # Reindex to align with expected columns
    features_df = features_df.reindex(columns=expected_columns, fill_value=0.0)
    
    return features_df


def preprocess_raw_data(raw_data, sensor_names=None, n_fft=DEFAULT_N_FFT, 
                        align_to_training=True):
    """
    Complete preprocessing pipeline for raw sensor data.
    
    Parameters
    ----------
    raw_data : np.ndarray
        Raw sensor data of shape (n_samples, n_timesteps, n_sensors) or 
        (n_timesteps, n_sensors) for a single sample.
    
    sensor_names : list of str, optional
        Sensor names. If None, uses the combined configuration.
    
    n_fft : int
        Number of FFT components to extract.
    
    align_to_training : bool
        If True, aligns features to the standard training format.
    
    Returns
    -------
    pd.DataFrame
        Preprocessed features ready for model input.
    """
    if sensor_names is None:
        sensor_names = SENSOR_CONFIG_COMBINED
    
    # Handle single sample case
    if raw_data.ndim == 2:
        raw_data = raw_data[np.newaxis, :, :]
    
    # Extract features
    features_df = extract_features(raw_data, sensor_names, n_fft=n_fft)
    
    # Align to training format if requested
    if align_to_training:
        expected_columns = build_expected_columns(sensor_names, n_fft)
        features_df = align_features_with_training(features_df, expected_columns)
    
    return features_df


def load_and_preprocess_npy(file_path, observation_index=None, sensor_names=None, 
                            n_fft=DEFAULT_N_FFT):
    """
    Load a .npy file and preprocess it.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the .npy file.
    
    observation_index : int, optional
        Index of the specific observation to preprocess. 
        If None, preprocesses all observations.
    
    sensor_names : list of str, optional
        Sensor names. If None, uses the combined configuration.
    
    n_fft : int
        Number of FFT components.
    
    Returns
    -------
    pd.DataFrame
        Preprocessed features.
    """
    # Load data
    data = np.load(file_path)
    
    # Extract specific observation if requested
    if observation_index is not None:
        data = data[observation_index]
    
    # Preprocess
    features_df = preprocess_raw_data(data, sensor_names, n_fft)
    
    return features_df


def get_feature_info(n_fft=DEFAULT_N_FFT, sensor_names=None):
    """
    Get information about the feature space.
    
    Returns a dictionary with feature statistics useful for model debugging.
    """
    if sensor_names is None:
        sensor_names = SENSOR_CONFIG_COMBINED
    
    n_sensors = len(sensor_names)
    features_per_sensor = 2 + n_fft  # mean + std + n_fft components
    total_features = n_sensors * features_per_sensor
    
    return {
        'n_sensors': n_sensors,
        'n_fft': n_fft,
        'features_per_sensor': features_per_sensor,
        'total_features': total_features,
        'sensor_names': sensor_names,
    }


# Constants for the model
N_FFT_DEFAULT = DEFAULT_N_FFT
EXPECTED_FEATURES = len(build_expected_columns(SENSOR_CONFIG_COMBINED, DEFAULT_N_FFT))
