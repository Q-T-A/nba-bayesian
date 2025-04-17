import matplotlib

matplotlib.use("Agg")
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy.stats import shapiro, normaltest, ttest_1samp, ttest_ind,skew, kurtosis, norm, anderson
from scipy.optimize import minimize
import numpy as np
import os


def create_output_dirs(model, dataset, algo):
    os.makedirs(f"outputs/{algo}/{dataset}/{model}", exist_ok=True)
    os.makedirs(f"stds/{algo}/{dataset}", exist_ok=True)
    os.makedirs(f"models/{dataset}", exist_ok=True)

def log_likelihood(params, data):
    mu, sigma = params[0], params[1]
    # We avoid log(0) by ensuring sigma > 0 (in case of numerical instability)
    return -np.sum(np.log(norm.pdf(data, mu, sigma)))
def actual_vs_predicted(y_test, y_pred, model, dataset, algo):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color="blue")
    plt.plot(
        [min(y_test), max(y_test)],
        [min(y_test), max(y_test)],
        color="red",
        linestyle="--",
    )
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.savefig(f"outputs/{algo}/{dataset}/{model}/actual_vs_predicted.png")
    plt.close()


def residuals(y_test, y_pred, model, dataset, algo):
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=30, color="skyblue")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution")
    plt.savefig(f"outputs/{algo}/{dataset}/{model}/residual_distribution.png")
    plt.close()


def feature_importances(importances, features, model, dataset, algo):
    feature_importance_df = pd.DataFrame(
        {"Feature": features, "Importance": importances}
    )

    feature_importance_df = feature_importance_df.sort_values(
        by="Importance", ascending=False
    ).head(20)

    plt.figure(figsize=(10, 6))
    plt.barh(
        feature_importance_df["Feature"][::-1],
        feature_importance_df["Importance"][::-1],
        color="skyblue",
    )
    plt.xlabel("Feature Importance")
    plt.title("Top 20 Feature Importances")
    plt.savefig(f"outputs/{algo}/{dataset}/{model}/best_features.png")
    plt.close()


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
    # Initial guess for mean (mu) and standard deviation (sigma)
    initial_guess = [np.mean(residuals), np.std(residuals)]

    # Minimize the negative log-likelihood to find the best parameters
    result = minimize(log_likelihood, initial_guess, args=(residuals,), bounds=[(None, None), (1e-6, None)])

    # Get the optimized parameters
    mu_mle, sigma_mle = result.x

    # Display the results
    f.write(f"Maximum Likelihood Estimate for Mean (mu): {mu_mle}\n")
    f.write(f"Maximum Likelihood Estimate for Std Dev (sigma): {sigma_mle}\n")
    result = anderson(residuals)

    # print the Anderson-Darling statistic and critical values for different significance levels
    f.write(f'Anderson-Darling statistic: {result.statistic}\n')
    f.write(f'Critical values: {result.critical_values}\n')
    f.write(f'Significance levels: {result.significance_level}\n')
        

NBA_TEAMS = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards"
}