import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)
# Load the data
data = pd.read_parquet(f"datasets/q1_q2.parquet")
data = data[8000:]
X = data.drop(["HFINAL", "AFINAL"], axis=1)
X = X.sort_index(axis=1)
X["HOME"] = X["HOME"].astype("category")
X["AWAY"] = X["AWAY"].astype("category")

y = (data["HFINAL"] > data["AFINAL"]).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

xgb_classifier = xgb.XGBClassifier(
        max_depth=4,
        learning_rate=0.01,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        enable_categorical=True,
    )

LAMBDA = 0.01
years = X_train["GAME_ID"].apply(lambda x: int(x[2:5]))
weights = np.exp(-LAMBDA * (years.max() - years))

xgb_classifier.fit(
        X_train.drop(["GAME_ID"], axis=1), y_train, sample_weight=weights
    )

print("WINNER=====")
y_pred = xgb_classifier.predict(X_test.drop(["GAME_ID"], axis=1))
y_pred_proba = xgb_classifier.predict_proba(X_test.drop(["GAME_ID"], axis=1))[
            :, 1
        ]  # Probability of home win            
leader_proba = y_pred_proba
# Convert to numpy arrays for easier handling
leader_proba = np.array(leader_proba)
y_test = np.array(y_test)

# Compute the Brier Score (already done)
brier_score = brier_score_loss(y_test, leader_proba)
print(f"Brier Score: {brier_score}")

# Calibration curve
fraction_of_positives, mean_predicted_value = calibration_curve(y_test, leader_proba, n_bins=10)

# Plot the calibration curve
plt.figure(figsize=(8, 6))
plt.plot(mean_predicted_value, fraction_of_positives, marker='o', label="Model Calibration")
plt.plot([0, 1], [0, 1], linestyle='--', label="Perfectly calibrated", color='gray')
plt.title("Calibration Curve q1_q2 winner")
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives")
plt.legend()
plt.grid(True)
plt.savefig('calibration_curve_q1_q2')
