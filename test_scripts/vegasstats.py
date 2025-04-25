import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy.stats import shapiro, normaltest, ttest_1samp, ttest_ind,skew, kurtosis, norm, anderson
from scipy.optimize import minimize
import numpy as np
import os


def eval_metrics(y_test, y_pred, f):
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = mse**0.5
    f.write(f"Test Set MSE: {mse:.3f}\n")
    f.write(f"Test Set RMSE: {rmse:.3f}\n")
    f.write(f"Test Set R²: {r2:.3f}\n")
    residuals = y_test - y_pred

    # Shapiro-Wilk Test for normality
    stat, p_value_shapiro = shapiro(residuals)
    f.write(f"Shapiro-Wilk Test p-value: {p_value_shapiro}\n")

    # T-test to check if residuals mean is significantly different from 0
    t_stat, p_value_ttest = ttest_ind(y_pred, y_test)
    f.write(f"T-test p-value: {p_value_ttest}\n")

    skewness = skew(residuals)
    kurt = kurtosis(residuals)

    f.write(f'Skewness: {skewness}\n')
    f.write(f'Kurtosis: {kurt}\n')
odds = pd.read_parquet('datasets/odds.parquet')
odds = odds.dropna()
totals_v = odds['OVERUNDER_Q2']
real = pd.read_parquet('datasets/q2/total.parquet')
combined = odds[['OVERUNDER_Q2']].join(real[['HFINAL', 'AFINAL']], how='inner')


combined['TOTAL_POINTS'] = combined['HFINAL'] + combined['AFINAL']

over_under = combined['OVERUNDER_Q2'].values
actual_points = combined['TOTAL_POINTS'].values
with open("outputs/book_line_eval.log", "w") as f:
    f.write("Evaluation of Bookmaker Over/Under Lines vs Actual Totals\n")
    f.write("---------------------------------------------------------\n")
    eval_metrics(actual_points, over_under, f)

