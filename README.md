# ✈️ Flight Delay Prediction System

## 🌟 Project Overview

This project implements a deployable, machine learning-driven system to predict the **arrival delay rate** (proportion of flights delayed $\ge 15$ minutes) for specific airline-airport routes.

The pipeline is structured following **MLOps best practices**, featuring **hyperparameter optimization** using time-series validation and a simulated **deployment service** with **data drift monitoring**.

### Key Features & Technologies

| Category | Component | Data Scientist Level Showcase |
| :--- | :--- | :--- |
| **Prediction Target** | Arrival Delay Rate ($0.0$ to $1.0$) | Regression problem using highly granular data. |
| **Model** | Optimized Random Forest Regressor | High-performance ensemble model. |
| **Feature Engineering**| **Target Encoding** | Advanced handling of high-cardinality categorical variables (`carrier`, `airport`). |
| **Robustness** | **Time-Series Cross-Validation** | Prevents time-series data leakage during tuning. |
| **MLOps** | **Serialization (`joblib`, `pickle`)** | Separation of model, encoders, and statistics for production. |
| **Monitoring** | **Data Drift Detection** | MLOps logic to flag anomalies in incoming data volume. |
| **Technologies** | Python, Pandas, Scikit-learn, Matplotlib | Professional, production-ready stack. |

-----

## ⚙️ Repository Structure

The project is broken down into modular files for maintainability and clear separation of concerns:

```
flight_delay_predictor/
├── data/
│   └── Airline_Delay_Cause.csv   # Raw data source
├── output/
│   └── feature_importance_plot.png # Visual output
├── config.py                     # Global constants, paths, and hyperparameters
├── features.py                   # Data cleaning and feature engineering logic
├── model.py                      # Model definition, optimization, and evaluation
├── predict.py                    # Deployment service logic (inference + monitoring)
├── train.py                      # Orchestrator script for the training pipeline
├── carrier_encoder.pkl           # ENCODER ARTIFACT
├── airport_encoder.pkl           # ENCODER ARTIFACT
├── training_stats.pkl            # STATS ARTIFACT for MLOps monitoring
└── rf_delay_predictor_model.joblib # TRAINED MODEL ARTIFACT
```

-----

## 🚀 Getting Started

### Prerequisites

You need Python 3.8+ and the following libraries:

```bash
pip install pandas scikit-learn matplotlib joblib numpy
```

### 1\. Data Setup

1.  Create a directory named `data` in the root of the project.
2.  Place the `Airline_Delay_Cause.csv` file inside the `data` directory.
3.  Create an empty directory named `output` for saving visualizations.

### 2\. Training the Model (ML Pipeline Execution)

Run the `train.py` script to execute the entire pipeline:

1.  Load and clean the data.
2.  Create and save all artifacts (`.pkl` encoders, `training_stats.pkl`).
3.  Perform **Hyperparameter Optimization** using **Time-Series Cross-Validation**.
4.  Train the final, optimized model and save it (`.joblib`).
5.  Evaluate the model and save the Feature Importance plot.

<!-- end list -->

```bash
python train.py
```

### 3\. Making Real-Time Predictions (Deployment Mock-up)

The `predict.py` script simulates a prediction service. It loads the saved artifacts and uses them to make an inference on new data.

It also includes the **Data Drift Monitoring** logic, which will trigger an alert if the input air traffic volume deviates significantly from the training data mean.

```bash
python predict.py
```

-----

## 📊 Model Performance and Interpretability

The model was optimized using `RandomizedSearchCV` and validated using `TimeSeriesSplit` to ensure robustness.

### Key Performance Metrics (Example)

| Metric | Value (Post-Optimization) | Goal |
| :--- | :--- | :--- |
| **$R^2$ Score** | **\~0.41 - 0.45** | Explains variance in the delay rate. (Expected value to slightly improve upon the baseline $0.41$ after tuning). |
| **Mean Absolute Error (MAE)** | **\~0.06 - 0.07** | Average prediction error is around 6-7 percentage points. |

### Feature Importance

The model's interpretability confirms that **engineered features** are the primary drivers of prediction:

| Feature | Importance | Rationale |
| :--- | :--- | :--- |
| **`airport_encoded`** | \~41% | Historical delay trend of the destination airport is the single largest factor. |
| **`arr_flights`** | \~34% | High flight volume (congestion) directly impacts delay rate. |
| **`carrier_encoded`** | \~25% | Historical performance of the specific airline is a major contributor. |

-----

## 🛡️ MLOps: Data Drift Monitoring

The **`predict.py`** module includes a monitoring mechanism that calculates the Z-score for the `arr_flights` feature in real-time.

If the incoming traffic volume exceeds a pre-defined threshold (set to $Z > 3.0$ in `config.py`), a warning is logged:

```
🚨 MLOps Alert: Data Drift Detected in 'arr_flights' (Input: 5000)
  Z-Score: 13.59. Suggests model re-evaluation or re-training.
```
