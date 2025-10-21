# config.py

import os

# Define the base directory (where config.py, train.py, etc., live)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- File Paths ---
DATA_FILE = os.path.join(BASE_DIR, 'data', 'Airline_Delay_Cause.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'rf_delay_predictor_model.joblib')
CARRIER_ENCODER_PATH = os.path.join(BASE_DIR, 'carrier_encoder.pkl')
AIRPORT_ENCODER_PATH = os.path.join(BASE_DIR, 'airport_encoder.pkl')
TRAINING_STATS_PATH = os.path.join(BASE_DIR, 'training_stats.pkl') # NEW: To store mean/std for drift detection
VISUAL_PATH = os.path.join(BASE_DIR, 'output', 'feature_importance_plot.png')

# --- Features & Target ---
FEATURES = ['year', 'month', 'arr_flights', 'carrier_encoded', 'airport_encoded']
TARGET = 'delay_rate'
TEST_SIZE = 0.2

# --- Model Hyperparameters (Base) ---
MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': 42,
    'n_jobs': -1
}

# --- Hyperparameter Search Grid for Optimization (RandomizedSearchCV) ---
RANDOM_GRID = {
    'n_estimators': [100, 200, 500, 800],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
# Number of parameter settings that are sampled (reduce if tuning takes too long)
N_ITER_SEARCH = 10
CV_SPLITS = 5 # Number of splits for TimeSeriesSplit

# --- MLOps / Deployment Monitoring ---
GLOBAL_MEAN_DELAY = 0.282066 # Fallback mean delay rate
DRIFT_THRESHOLD = 3.0        # Z-score threshold for Data Drift alert