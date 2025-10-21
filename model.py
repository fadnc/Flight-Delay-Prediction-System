# model.py

import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from config import MODEL_PATH, MODEL_PARAMS, RANDOM_GRID, N_ITER_SEARCH, CV_SPLITS, VISUAL_PATH

def train_model(X_train, y_train):
    """Initializes, optimizes, and trains the Random Forest Regressor."""
    
    # 1. Base Model Initialization
    base_model = RandomForestRegressor(**MODEL_PARAMS)

    # 2. Time-Series Cross-Validation Setup (for time-aware splitting)
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    print(f"Using Time-Series Cross-Validation (n_splits={CV_SPLITS}) for robust tuning.")

    # 3. Hyperparameter Search (Randomized Search)
    rf_random = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=RANDOM_GRID,
        n_iter=N_ITER_SEARCH,
        cv=tscv,
        verbose=1, # Show optimization progress
        random_state=42,
        n_jobs=-1,
        scoring='neg_mean_absolute_error'
    )

    print("\n--- Starting Hyperparameter Optimization (Randomized Search) ---")
    rf_random.fit(X_train, y_train)
    print("Optimization Complete.")

    # 4. Save the Optimized Model
    optimized_model = rf_random.best_estimator_
    print(f"Best Parameters Found: {rf_random.best_params_}")
    
    joblib.dump(optimized_model, MODEL_PATH)
    print(f"Optimized Model saved to: {MODEL_PATH}")
    
    return optimized_model

def visualize_feature_importance(model, feature_names):
    """Generates and saves the feature importance bar plot."""
    
    feature_importance = pd.Series(
        model.feature_importances_, 
        index=feature_names
    ).sort_values(ascending=False).reset_index()

    feature_importance.columns = ['Feature', 'Importance']

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='#00796B')
    plt.xlabel("Feature Importance (Gini-based Score)")
    plt.title("Optimized Random Forest Feature Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(VISUAL_PATH), exist_ok=True) 
    plt.savefig(VISUAL_PATH)
    print(f"Feature importance plot saved to {VISUAL_PATH}")
    
    return feature_importance

def evaluate_model(model, X_test, y_test):
    """Evaluates the trained model and prints metrics."""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Optimized Model Performance Metrics ---")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared Score (R2): {r2:.4f}")

    # Call the visualization function
    visualize_feature_importance(model, X_test.columns) 
    
    return mae, r2