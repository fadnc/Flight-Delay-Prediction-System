# config.py

import os

# Define the base directory (where config.py, train.py, etc., live)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- File Paths ---
# This line is the critical fix: joining the BASE_DIR, the 'data' folder, and the file name.
DATA_FILE = os.path.join(BASE_DIR, 'data', 'Airline_Delay_Cause.csv')

# Artifact paths (ensure these directories exist before running train.py)
MODEL_PATH = os.path.join(BASE_DIR, 'rf_delay_predictor_model.joblib')
CARRIER_ENCODER_PATH = os.path.join(BASE_DIR, 'carrier_encoder.pkl')
AIRPORT_ENCODER_PATH = os.path.join(BASE_DIR, 'airport_encoder.pkl')

# --- Model Hyperparameters ---
MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': 42,
    'n_jobs': -1
}

# --- Features & Target ---
FEATURES = ['year', 'month', 'arr_flights', 'carrier_encoded', 'airport_encoded']
TARGET = 'delay_rate'
TEST_SIZE = 0.2

# --- Deployment / Fallback ---
GLOBAL_MEAN_DELAY = 0.282066