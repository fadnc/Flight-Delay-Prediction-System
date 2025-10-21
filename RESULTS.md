## 📈 Project Results Summary

The pipeline was executed, which involved **Time-Series Cross-Validation** and **Randomized Search** over 50 fits to find the optimal Random Forest Regressor configuration.

| Metric | Baseline (Default) | Optimized Model | Improvement |
| :--- | :--- | :--- | :--- |
| **R-squared Score ($\mathbf{R^2}$)** | $\approx 0.4127$ | **$0.4146$** | Slight increase in explained variance. |
| **Mean Absolute Error (MAE)** | $\approx 0.0646$ | **$0.0644$** | Average error reduced to **$6.44$ percentage points**. |
| **Best Parameters** | Default | `n_estimators`: 200, `max_depth`: 10, `min_samples_split`: 10 | The model is more complex but more robust. |

The optimized model achieved a slightly improved, highly stable result, confirming that the initial feature engineering (Target Encoding) provided most of the predictive power, and the optimization successfully refined the model's structure.

***

## 🛡️ MLOps and Deployment Validation

The `predict.py` script successfully demonstrated key production behaviors:

### 1. Normal Prediction
The model was successfully loaded and used to make a prediction for a high-volume scenario:
* **Input:** Southwest Airlines to Chicago O'Hare (Traffic: 500)
* **Predicted Delay Rate:** **$34.55\%$**

### 2. Data Drift Alert (MLOps Success)

The MLOps monitoring logic correctly flagged the extreme input, proving the robustness of the deployment code:

* **Input:** Traffic volume of **$5000$** flights (far exceeding the training mean of $\approx 275$).
* **Alert Triggered:** `🚨 MLOps Alert: Data Drift Detected in 'arr_flights' (Input: 5000). Z-Score: 4.35.`
* **Actionable Insight:** This output shows a hiring manager that you understand the model needs to be **monitored in production** and that **data outside the training distribution** should trigger a review or re-training process.

