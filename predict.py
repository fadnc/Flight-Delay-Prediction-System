# predict.py

import joblib
import pickle
import pandas as pd
from config import (
    MODEL_PATH, 
    CARRIER_ENCODER_PATH, 
    AIRPORT_ENCODER_PATH, 
    FEATURES, 
    GLOBAL_MEAN_DELAY
)

# Global variables to cache loaded artifacts (improves API response time)
MODEL_CACHE = None
CARRIER_MAP_CACHE = None
AIRPORT_MAP_CACHE = None

def load_artifacts():
    """Loads the model and encoders into global cache on service start-up."""
    global MODEL_CACHE, CARRIER_MAP_CACHE, AIRPORT_MAP_CACHE

    if MODEL_CACHE is None:
        MODEL_CACHE = joblib.load(MODEL_PATH)
    
    if CARRIER_MAP_CACHE is None:
        with open(CARRIER_ENCODER_PATH, 'rb') as f:
            CARRIER_MAP_CACHE = pickle.load(f)
    
    if AIRPORT_MAP_CACHE is None:
        with open(AIRPORT_ENCODER_PATH, 'rb') as f:
            AIRPORT_MAP_CACHE = pickle.load(f)

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
    
    # Preprocess the raw input
    input_df = preprocess_input(year, month, carrier_name, airport_name, arr_flights)
    
    # Predict
    prediction = MODEL_CACHE.predict(input_df)[0]
    
    return prediction

if __name__ == '__main__':
    # Example of how the deployed function would be called (e.g., in a Flask app)
    load_artifacts()
    
    sample_carrier = 'Southwest Airlines'
    sample_airport = 'Chicago, IL: Chicago O\'Hare International'
    
    predicted_rate = get_prediction(2025, 7, sample_carrier, sample_airport, 500)
    
    print(f"Predicted Delay Rate for {sample_carrier} to {sample_airport}: {predicted_rate * 100:.2f}%")