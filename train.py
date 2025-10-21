# train.py

from features import load_and_clean_data, create_and_save_artifacts, prepare_training_data
from model import train_model, evaluate_model

if __name__ == '__main__':
    print("--- 1. Data Loading and Cleaning ---")
    df = load_and_clean_data()
    print(f"Cleaned Data Shape: {df.shape}")
    
    print("\n--- 2. Creating & Saving Artifacts (Encoders, Stats) ---")
    carrier_map, airport_map = create_and_save_artifacts(df)
    
    print("\n--- 3. Preparing Data Splits ---")
    X_train, X_test, y_train, y_test = prepare_training_data(df, carrier_map, airport_map)
    print(f"Train/Test Split: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples.")

    # 4. Train and Optimize Model
    trained_model = train_model(X_train, y_train)
    
    # 5. Evaluate and Visualize
    evaluate_model(trained_model, X_test, y_test)