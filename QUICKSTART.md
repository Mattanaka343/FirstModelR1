# Quick Start Guide - Activity Classification Streamlit App

## Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

This will install:
- **streamlit**: The web app framework
- **numpy**: Numerical computing
- **pandas**: Data manipulation
- **scikit-learn**: Machine learning models

### 2. Verify Data Files
Check that your project structure has:
```
PrimerModeloReto1/
├── Data/
│   ├── TransformedData.csv          ✓
│   └── d01_raw_data/
│       ├── 000_1.npy, 000_2.npy
│       ├── 001_1.npy, 001_2.npy
│       └── ... (up to 015_1.npy, 015_2.npy)
└── Models/                           ✓ (created automatically)
```

---

## Option A: Quick Test (Without Training Your Own Model)

If you just want to test the app structure without a trained model:

```bash
streamlit run app.py
```

The app will work perfectly - you'll be able to:
- ✅ Simulate observations
- ✅ Preprocess data
- ⚠️  Model selection will show "No models available" (this is expected)

---

## Option B: Full Setup (With Your Own Trained Model)

### Step 1: Train and Save a Model
```bash
python Scripts/train_model_example.py
```

This script will:
1. Load `Data/TransformedData.csv`
2. Train a Random Forest classifier
3. Save it to `Models/random_forest_model.pkl`
4. Print accuracy metrics

Output example:
```
Training accuracy: 0.9234
Test accuracy: 0.8934

Model saved successfully!
File size: 2134.56 KB
```

### Step 2: Launch the Streamlit App
```bash
streamlit run app.py
```

### Step 3: Use the App
1. **Sidebar:** Select Activity (e.g., "015") and Model ("random_forest_model.pkl")
2. **Step 1:** Click "🎲 Simulate Random Observation"
3. **Step 2:** Click "Preprocess Data"
4. **Step 3:** Click "🎯 Make Prediction"

---

## Understanding the App Workflow

### Raw Data → Features → Predictions

```
.npy File (Time-series)
     ↓
[Extract]
- Mean per sensor
- Std Dev per sensor
- FFT components per sensor
     ↓
Feature Vector (144 dims)
     ↓
[Load Model]
- Random Forest (or your model)
     ↓
Activity Prediction (000-015)
```

---

## Preprocessing Pipeline

The app implements the same preprocessing from `FirstModel.qmd`:

**For each sensor, extract:**
- `{sensor}_mean`: Average value across time
- `{sensor}_std`: Variability across time
- `{sensor}_fft_1` to `{sensor}_fft_10`: Frequency components

**Total sensors:** 12 (pitch1, yaw1, roll1, pitch2, yaw2, roll2, f1, f2, f3, f4, f5, pitch3)

**Total features:** 144 (12 sensors × 12 features each)

---

## Configuration Options

In the Streamlit sidebar, you can adjust:

| Option | Effect | Default |
|--------|--------|---------|
| **Select Activity** | Which activity to simulate | First activity |
| **Select Model** | Which model to use for prediction | First available |
| **Number of FFT Components** | Feature dimensionality | 10 |

⚠️ **Important:** If you change FFT components, your model must expect that feature dimension!

---

## Training Your Own Model

### Using Scikit-learn
```python
from sklearn.ensemble import RandomForestClassifier
import pickle
from Scripts.preprocessing_utils import EXPECTED_FEATURES

# After loading and preprocessing your data
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Save for the app
with open('Models/my_model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

The model should:
- Accept input shape: `(n_samples, 144)` or `(144,)`
- Implement `.predict()` method
- Return activity class (000-015)

### Using the Reusable Preprocessing Utilities
```python
from Scripts.preprocessing_utils import preprocess_raw_data
import numpy as np

# Load raw data
raw_data = np.load('Data/d01_raw_data/000_1.npy')

# Preprocess
features = preprocess_raw_data(raw_data)  # Returns DataFrame

# Use with your model
prediction = model.predict(features.values)
```

---

## Troubleshooting

### Q: "No activity data found" error
**A:** Verify the path to raw data:
```bash
ls Data/d01_raw_data/
# Should show: 000_1.npy, 000_2.npy, 001_1.npy, etc.
```

### Q: "No trained models found" warning
**A:** This is normal on first run. Train a model:
```bash
python Scripts/train_model_example.py
```

### Q: Prediction accuracy is low
**A:** Make sure:
1. Model was trained on same preprocessing (FFT=10)
2. Model was trained on `Data/TransformedData.csv`
3. Check model accuracy from training script
4. Verify input features match: 144 dimensions

### Q: "Error loading model" 
**A:** File might be corrupted. Retrain:
```bash
rm Models/random_forest_model.pkl
python Scripts/train_model_example.py
```

### Q: How do I stop the app?
**A:** Press `Ctrl+C` in the terminal

---

## Example Workflow

```bash
# Terminal 1: Install and setup
pip install -r requirements.txt
python Scripts/train_model_example.py

# Terminal 1: Launch app
streamlit run app.py

# Browser: Navigate to http://localhost:8501
# - Select Activity "015"
# - Click "Simulate Random Observation"
# - Click "Preprocess Data"
# - Select model "random_forest_model.pkl"
# - Click "Make Prediction"
# - See if prediction matches activity ID
```

---

## Project Structure

```
PrimerModeloReto1/
│
├── app.py                              # Main Streamlit application
├── requirements.txt                    # Python dependencies
├── QUICKSTART.md                      # This file
├── STREAMLIT_APP_README.md            # Detailed documentation
│
├── Data/
│   ├── TransformedData.csv            # Preprocessed training data
│   └── d01_raw_data/
│       └── {000-015}_{1,2}.npy        # Raw sensor files
│
├── Models/                             # Trained model directory (auto-created)
│   └── random_forest_model.pkl        # Example model
│
├── Scripts/
│   ├── preprocessing_utils.py         # Reusable preprocessing functions
│   ├── train_model_example.py         # Example training script
│   ├── model.py                       # Your existing model code
│   └── ...
│
├── Notebooks/
│   ├── FirstModel.qmd                 # Reference for preprocessing
│   ├── SecondModel.qmd
│   └── ...
│
└── Figures/, Reports/, SVM/, RandomForest/
    └── ... (other project files)
```

---

## Next Steps

1. **Run the quick test** → `streamlit run app.py`
2. **Train a model** → `python Scripts/train_model_example.py`
3. **Make predictions** → Use the Streamlit interface
4. **Customize preprocessing** → Edit `Scripts/preprocessing_utils.py`
5. **Integrate your own model** → Save as `.pkl` in `Models/` directory

---

## For More Information

- **Preprocessing details:** See [STREAMLIT_APP_README.md](STREAMLIT_APP_README.md)
- **FirstModel reference:** See [Notebooks/FirstModel.qmd](Notebooks/FirstModel.qmd)
- **Example training:** See [Scripts/train_model_example.py](Scripts/train_model_example.py)
- **Reusable utilities:** See [Scripts/preprocessing_utils.py](Scripts/preprocessing_utils.py)

---

**Happy modeling! 🚀**
