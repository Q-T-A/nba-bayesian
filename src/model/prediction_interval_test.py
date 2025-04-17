import matplotlib

matplotlib.use("Agg")
import json
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import numpy as np
import click
from utils import (
    actual_vs_predicted,
    residuals,
    feature_importances,
    eval_metrics,
    create_output_dirs,
)

import json
from scipy.stats import t
import numpy as np


def create_regressor():
    xgb_regressor = xgb.XGBRegressor(
        max_depth=4,
        learning_rate=0.01,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror",
        enable_categorical=True,
    )

    return xgb_regressor


def total_model(output, ev):
    data = pd.read_parquet("output.parquet")

    X = data.drop(["HFINAL", "AFINAL"], axis=1)
    X["HOME"] = X["HOME"].astype("category")
    X["AWAY"] = X["AWAY"].astype("category")

    y = data["HFINAL"] + data["AFINAL"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    xgb_regressor = create_regressor()

    # Weight calculations
    LAMBDA = 0.01
    years = X_train["GAME_ID"].apply(lambda x: int(x[2:5]))
    weights = np.exp(-LAMBDA * (years.max() - years))

    # Fit the model
    xgb_regressor.fit(X_train.drop(["GAME_ID"], axis=1), y_train, sample_weight=weights)

    if ev:
        print("TOTAL======")
        y_pred = xgb_regressor.predict(X_test.drop(["GAME_ID"], axis=1))

        # Calculate residuals and SE
        residual = y_test - y_pred
        n = len(y_test)
        k = X_test.shape[1]  # Number of predictors
        se = np.sqrt(np.sum(residual**2) / (n - k))  # Standard error

        # Get t* for 95% confidence interval
        t_star = t.ppf(0.975, df=n - k)  # Two-tailed
        print(t_star)
        # Prediction intervals
        lower_bound = y_pred - t_star * se
        upper_bound = y_pred + t_star * se

        create_output_dirs("total")
        with open("interval_test.log", "w") as f:
            eval_metrics(y_test, y_pred, f)
        feature_importances(
            xgb_regressor.feature_importances_,
            X.drop(["GAME_ID"], axis=1).columns,
            "total",
        )
        residuals(y_test, y_pred, "total")
        actual_vs_predicted(y_test, y_pred, "total")
        print("===========\n")

        # Save prediction intervals
        intervals = pd.DataFrame({
            "Prediction": y_pred,
            "Lower Bound": lower_bound,
            "Upper Bound": upper_bound,
        })
        intervals.to_csv("total_prediction_intervals.csv", index=False)

    if output:
        y_pred = xgb_regressor.predict(X_test.drop(["GAME_ID"], axis=1))
        residual = y_pred - y_test
        std = np.std(residual)
        with open('total_std.json', 'w') as f:
            json.dump({'std': std}, f)
        xgb_regressor.save_model("models/model.ubj")
total_model(False, True)