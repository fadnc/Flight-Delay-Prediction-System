# predict.py

import joblib
import pickle
import pandas as pd
import numpy as np
from config import (
    MODEL_PATH, 
    CARRIER_ENCODER_PATH, 
    AIRPORT_ENCODER_PATH, 
    TRAINING_STATS_PATH,
    FEATURES, 
    GLOBAL_MEAN_DELAY,
    DRIFT_THRESHOLD
)

# Global variables to cache loaded artifacts (improves API response time)
MODEL_CACHE = None
CARRIER_MAP_CACHE = None
AIRPORT_MAP_CACHE = None
STATS_CACHE = None

def load_artifacts():
    """Loads the model, encoders, and training statistics into global cache."""
    global MODEL_CACHE, CARRIER_MAP_CACHE, AIRPORT_MAP_CACHE, STATS_CACHE

    if MODEL_CACHE is None:
        MODEL_CACHE = joblib.load(MODEL_PATH)
    
    if CARRIER_MAP_CACHE is None:
        with open(CARRIER_ENCODER_PATH, 'rb') as f:
            CARRIER_MAP_CACHE = pickle.load(f)
    
    if AIRPORT_MAP_CACHE is None:
        with open(AIRPORT_ENCODER_PATH, 'rb') as f:
            AIRPORT_MAP_CACHE = pickle.load(f)

    if STATS_CACHE is None:
        with open(TRAINING_STATS_PATH, 'rb') as f:
            STATS_CACHE = pickle.load(f)

def check_for_drift(feature_value, feature_name):
    """Checks if a feature value is statistically too far from the training mean."""
    
    mean = STATS_CACHE[f'{feature_name}_mean']
    std = STATS_CACHE[f'{feature_name}_std']
    
    # Calculate Z-score
    if std == 0:
         z_score = 0
    else:
        z_score = abs(feature_value - mean) / std
    
    if z_score > DRIFT_THRESHOLD:
        print(f"\n🚨 MLOps Alert: Data Drift Detected in '{feature_name}' (Input: {feature_value})")
        print(f"  Z-Score: {z_score:.2f}. Suggests model re-evaluation or re-training.")
        return True
    return False

def preprocess_input(year, month, carrier_name, airport_name, arr_flights):
    """Applies the same feature engineering steps as used during training."""
    
    # Load artifacts if not already in cache
    load_artifacts()

    # 1. Target Encoding Lookup (with fallback for unseen categories)
    carrier_encoded = CARRIER_MAP_CACHE.get(carrier_name, GLOBAL_MEAN_DELAY)
    airport_encoded = AIRPORT_MAP_CACHE.get(airport_name, GLOBAL_MEAN_DELAY)

    # 2. Format as a DataFrame for the model
    input_data = pd.DataFrame([[year, month, arr_flights, carrier_encoded, airport_encoded]],
                              columns=FEATURES)
    
    return input_data

def get_prediction(year, month, carrier_name, airport_name, arr_flights):
    """The core function for making a prediction (the API endpoint logic)."""
    
    # MLOps Check: Monitor for Data Drift on arr_flights
    check_for_drift(arr_flights, 'arr_flights') 

    # Preprocess the raw input
    input_df = preprocess_input(year, month, carrier_name, airport_name, arr_flights)
    
    # Predict
    prediction = MODEL_CACHE.predict(input_df)[0]
    
    return prediction

if __name__ == '__main__':
    # Example 1: Normal Prediction
    load_artifacts()
    sample_carrier = 'Southwest Airlines'
    sample_airport = 'Chicago, IL: Chicago O\'Hare International'
    
    predicted_rate_normal = get_prediction(2025, 7, sample_carrier, sample_airport, 500)
    
    print("\n--- Deployment Service Mock-up ---")
    print(f"Normal Scenario Prediction (Traffic: 500): {predicted_rate_normal * 100:.2f}%")

    # Example 2: Prediction that triggers Drift Alert (using a very high traffic volume)
    predicted_rate_drift = get_prediction(2025, 7, sample_carrier, sample_airport, 5000)
    print(f"Drift Scenario Prediction (Traffic: 5000): {predicted_rate_drift * 100:.2f}%")