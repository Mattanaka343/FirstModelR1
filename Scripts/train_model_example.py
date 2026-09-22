"""
Example script showing how to train and save a model compatible with the Streamlit app.

This example demonstrates:
1. Loading preprocessed data from TransformedData.csv
2. Training a simple model (Random Forest or any scikit-learn model)
3. Saving the model in the correct format (.pkl) for the Streamlit app

Run this script to generate a model file in the Models/ directory.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path


def train_and_save_model(data_path='Data/TransformedData.csv', 
                        model_save_path='Models/random_forest_model.pkl'):
    """
    Train a Random Forest model and save it for use with the Streamlit app.
    """
    
    print("📖 Loading preprocessed data...")
    df = pd.read_csv(data_path)
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()[:5]}... (showing first 5)")
    
    # Separate features and labels
    X = df.drop('activity', axis=1).values
    y = df['activity'].values
    
    print(f"\n✅ Features shape: {X.shape}")
    print(f"✅ Labels shape: {y.shape}")
    print(f"✅ Unique activities: {np.unique(y)}")
    
    # Split the data
    print("\n📊 Splitting data (80/20 train/test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    
    # Train the model
    print("\n🤖 Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    print("✅ Model training complete!")
    
    # Evaluate
    print("\n📈 Evaluating model...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred))
    
    # Save the model
    print(f"\n💾 Saving model to {model_save_path}...")
    
    # Create parent directory if it doesn't exist
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(model_save_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"✅ Model saved successfully!")
    print(f"📦 File size: {Path(model_save_path).stat().st_size / 1024:.2f} KB")
    
    # Verify the model can be loaded
    print("\n🔍 Verifying model can be loaded...")
    with open(model_save_path, 'rb') as f:
        loaded_model = pickle.load(f)
    
    # Test prediction
    test_prediction = loaded_model.predict(X_test[:1])
    print(f"✅ Model loaded successfully!")
    print(f"✅ Sample prediction: {test_prediction[0]}")
    
    return model


if __name__ == "__main__":
    print("=" * 60)
    print("Model Training and Saving Example")
    print("=" * 60)
    
    # Train and save the model
    train_and_save_model()
    
    print("\n" + "=" * 60)
    print("✅ Complete! Your model is ready for the Streamlit app")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run: streamlit run app.py")
    print("2. Select the model 'random_forest_model.pkl' in the sidebar")
    print("3. Simulate observations and make predictions!")
