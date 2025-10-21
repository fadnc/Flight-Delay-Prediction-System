# train.py

from features import load_and_clean_data, create_and_save_encoders, prepare_training_data
from model import train_model, evaluate_model

if __name__ == '__main__':
    # 1. Load and clean data
    df = load_and_clean_data()
    
    # 2. Create and save encoding artifacts
    carrier_map, airport_map = create_and_save_encoders(df)
    
    # 3. Prepare training and testing splits
    X_train, X_test, y_train, y_test = prepare_training_data(df, carrier_map, airport_map)
    
    # 4. Train model and save the artifact
    trained_model = train_model(X_train, y_train)
    
    # 5. Evaluate
    evaluate_model(trained_model, X_test, y_test)