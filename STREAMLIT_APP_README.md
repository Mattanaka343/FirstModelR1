# Activity Classification Streamlit App

This application provides an interactive interface to simulate sensor observations, preprocess them, and make activity predictions using trained machine learning models.

## Overview

The app consists of three main stages:

### 1. **Simulate Observation** 🎲
- Select an activity (000-015)
- Randomly selects a sensor observation from the activity's raw data files
- Displays the simulated raw sensor data shape and details

### 2. **Preprocess Data** 🔧
- Applies the same preprocessing pipeline used in `FirstModel.qmd`
- Extracts features using:
  - **Mean** of each sensor signal
  - **Standard Deviation** of each sensor signal
  - **FFT Components** (configurable, default 10) of each sensor signal
- Aligns features to match the training data format
- Produces a feature vector ready for model input

### 3. **Make Prediction** 🎯
- Loads a trained model from the `Models/` directory
- Feeds preprocessed features to the model
- Displays the predicted activity and compares it to ground truth

## Setup

### Prerequisites
```bash
pip install streamlit numpy pandas scikit-learn
```

### File Structure
```
PrimerModeloReto1/
├── app.py                          # Main Streamlit application
├── Data/
│   ├── TransformedData.csv        # Preprocessed training data
│   └── d01_raw_data/              # Raw .npy sensor files (000_1.npy, 000_2.npy, etc.)
├── Models/                         # Directory for trained model files (.pkl)
├── Notebooks/
│   ├── FirstModel.qmd             # Contains preprocessing logic
│   └── ...
└── Scripts/
    └── ...
```

## Running the App

From the project root directory, run:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## How to Add a Trained Model

1. Train your model using the preprocessing pipeline defined in `FirstModel.qmd`
2. Save the trained model as a pickle file (`.pkl`):

```python
import pickle

# After training your model
with open('Models/your_model_name.pkl', 'wb') as f:
    pickle.dump(trained_model, f)
```

3. The model must implement a `predict()` method that accepts:
   - Input shape: `(1, num_features)` or `(num_features,)`
   - Should return a class label (activity ID)

## Sensor Configuration

The app uses a combined sensor configuration with two groups:

**Sensor Group 1** (from _1 files):
- pitch1, yaw1, roll1, pitch2, yaw2, roll2 (6 sensors)

**Sensor Group 2** (from _2 files + additional):
- f1, f2, f3, f4, f5, pitch3 (6 sensors)

**Total:** 12 sensors


### Feature Extraction

For each sensor, the following features are extracted:

- **1 mean value**
- **1 standard deviation value**
- **N FFT components** (default 10)

**Total features per sensor:** 12 (default)
**Total features overall:** 144 features (default with 12 sensors)

## Configuration Options

In the Streamlit sidebar, you can adjust:

- **Select Activity:** Choose which activity class to simulate from
- **Select Model:** Choose which trained model to use for prediction
- **Number of FFT Components:** Adjust the number of frequency components to extract (affects feature dimension)

## Preprocessing Details

The preprocessing pipeline:

1. **Load raw data:** `.npy` files containing time-series sensor readings
2. **Extract statistics:** Compute mean and standard deviation across the time dimension
3. **Extract frequency components:** Apply FFT and keep the first N magnitude components
4. **Create feature vector:** Concatenate all features into a single vector
5. **Align with training data:** Ensure features match the training data format and column order
6. **Fill missing values:** Any missing features are filled with 0.0

## Example Workflow

1. Run `streamlit run app.py`
2. Select Activity "015" from the sidebar
3. Click the **"🎲 Simulate Random Observation"** button
4. Review the raw data shape and preview
5. Click **"Preprocess Data"** button
6. Review the extracted features
7. Select a trained model from the sidebar
8. Click **"🎯 Make Prediction"** button
9. See the predicted activity compared to ground truth

## Troubleshooting

### No activities appear in dropdown
- Check that `Data/d01_raw_data/` directory exists and contains `.npy` files
- Verify files follow the naming convention: `XXX_1.npy` and `XXX_2.npy`

### No models appear in dropdown
- Check that `Models/` directory exists
- Ensure trained models are saved as `.pkl` files
- Verify pickle files contain a model with a `predict()` method

### Prediction errors
- Ensure your model was trained on data preprocessed with the same sensor configuration
- Verify FFT components match between preprocessing and training
- Check that your model's input expects the correct feature dimension (144 with default 10 FFT components)

## Integration with Your Workflow

This app complements your existing code:

- **Raw data loading** mimics the logic in `FirstModel.qmd`
- **Feature extraction** uses the same `extract_features()` function
- **Preprocessing** follows the same steps as your notebook
- **Model integration** expects pickle-saved models compatible with scikit-learn or similar

## Notes

- The app maintains session state to avoid reprocessing data unnecessarily
- Predictions are compared against the selected activity's ground truth
- The interface is designed for exploration and validation of model predictions
