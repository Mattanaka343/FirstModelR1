"""
Configuration file for the Activity Classification application.

Define all constants and settings in one place for easy maintenance.
"""

# Data paths
DATA_DIR = "Data"
RAW_DATA_DIR = "Data/d01_raw_data"
TRANSFORMED_DATA_FILE = "Data/TransformedData.csv"
MODELS_DIR = "Models"

# Sensor configuration
SENSOR_CONFIG_1 = [
    "pitch1",
    "yaw1", 
    "roll1",
    "pitch2",
    "yaw2",
    "roll2"
]

SENSOR_CONFIG_2 = [
    "f1",
    "f2",
    "f3",
    "f4",
    "f5",
    "pitch3"
]

SENSOR_CONFIG_COMBINED = SENSOR_CONFIG_1 + SENSOR_CONFIG_2

# Preprocessing parameters
DEFAULT_N_FFT = 10
MIN_FFT_COMPONENTS = 1
MAX_FFT_COMPONENTS = 20

# Feature extraction
EXPECTED_N_SENSORS = len(SENSOR_CONFIG_COMBINED)
FEATURES_PER_SENSOR = 2 + DEFAULT_N_FFT  # mean + std + fft components
EXPECTED_TOTAL_FEATURES = EXPECTED_N_SENSORS * FEATURES_PER_SENSOR

# Activity information
ACTIVITY_IDS = [str(i).zfill(3) for i in range(16)]  # 000-015

# Streamlit app settings
APP_TITLE = "Activity Classification Interface"
APP_PAGE_ICON = "🏃"
APP_LAYOUT = "wide"

# Model settings
MODEL_FILE_EXTENSION = ".pkl"
DEFAULT_MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 15,
    'random_state': 42,
    'n_jobs': -1,
}

# UI/UX settings
SHOW_RAW_DATA_PREVIEW = True
SHOW_FEATURES = True
SHOW_MORE_INFO = True
SAMPLE_PREVIEW_ROWS = 5

# Display precision
FLOAT_PRECISION = 4
ACCURACY_PRECISION = 4
