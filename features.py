import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from config import DATA_FILE, CARRIER_ENCODER_PATH, AIRPORT_ENCODER_PATH, TARGET, FEATURES, TEST_SIZE, TRAINING_STATS_PATH

def load_and_clean_data():
    """Loads, cleans, and prepares the base dataframe."""
    df = pd.read_csv(DATA_FILE)
    df.dropna(subset=['arr_del15'], inplace=True)
    df = df[df['arr_flights'] > 0].copy()
    df['delay_rate'] = df['arr_del15'] / df['arr_flights']
    return df

def create_and_save_artifacts(df):
    """Calculates and serializes encoding mappings and training stats."""
    
    #Target Encoding mappings
    carrier_mapping = df.groupby('carrier_name')[TARGET].mean().to_dict()
    airport_mapping = df.groupby('airport_name')[TARGET].mean().to_dict()

    with open(CARRIER_ENCODER_PATH, 'wb') as f:
        pickle.dump(carrier_mapping, f)
    with open(AIRPORT_ENCODER_PATH, 'wb') as f:
        pickle.dump(airport_mapping, f)
        
    print(f"Encoders saved to {CARRIER_ENCODER_PATH} and {AIRPORT_ENCODER_PATH}")

    # Training statistics
    training_stats = {
        'arr_flights_mean': df['arr_flights'].mean(),
        'arr_flights_std': df['arr_flights'].std()
    }
    with open(TRAINING_STATS_PATH, 'wb') as f:
        pickle.dump(training_stats, f)
    
    print(f"Training statistics saved to {TRAINING_STATS_PATH}")

    return carrier_mapping, airport_mapping

def prepare_training_data(df, carrier_mapping, airport_mapping):
    """Applies encoding and splits data for training."""
    
    # Apply mappings to create encoded features
    df['carrier_encoded'] = df['carrier_name'].map(carrier_mapping)
    df['airport_encoded'] = df['airport_name'].map(airport_mapping)

    X = df[FEATURES]
    y = df[TARGET]
    
    # Perform  split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42
    )

    return X_train, X_test, y_train, y_test
