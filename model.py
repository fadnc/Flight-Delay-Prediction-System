# model.py

import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from config import MODEL_PATH, MODEL_PARAMS

def train_model(X_train, y_train):
    """Initializes and trains the Random Forest Regressor."""
    model = RandomForestRegressor(**MODEL_PARAMS)
    print("Training Random Forest Regressor...")
    model.fit(X_train, y_train)
    print("Training Complete.")
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")
    return model

def visualize_feature_importance(model, feature_names):
    """Generates and saves the feature importance bar plot."""
    
    # Extract importance scores
    feature_importance = pd.Series(
        model.feature_importances_, 
        index=feature_names
    ).sort_values(ascending=False).reset_index()

    feature_importance.columns = ['Feature', 'Importance']

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='#00796B')
    plt.xlabel("Feature Importance (Gini-based Score)")
    plt.title("Random Forest Feature Importance for Flight Delay Rate Prediction")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    # Save the plot (e.g., to an 'output' folder)
    plt.savefig("output/feature_importance_plot.png")
    print("Feature importance plot saved to output/.")
    
    return feature_importance

def evaluate_model(model, X_test, y_test):
    """Evaluates the trained model and prints metrics."""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Model Performance Metrics ---")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared Score (R2): {r2:.4f}")

    # Call the visualization function here
    # Assuming X_test columns are the feature names
    feature_importance_df = visualize_feature_importance(model, X_test.columns) 
    
    return mae, r2