import concurrent.futures
import json
import pandas as pd
import xgboost as xgb
import numpy as np
import click
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix, 
    classification_report
)
from sklearn.linear_model import LinearRegression
from joblib import dump
from utils import (
    actual_vs_predicted, 
    residuals, 
    feature_importances,
    eval_metrics, 
    create_output_dirs
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns


BACKTEST_CUTOFF = 12500

@click.command()
@click.option("-m", "--model",        default="all", type=click.Choice(["total", "winner", "spread", "homepoints", "awaypoints", "all"]))
@click.option("-d", "--dataset",      default="all", type=click.Choice(["pregame", "q1", "q2", "q3", "q1_q2", "q1_q2_q3", "q1_m", "q1_q2_m", "q1_q2_q3_m", "q1.o", "q2.o", "q3.o", "all"]))
@click.option("-a", "--algo", default="xgb", type=click.Choice(["linear", "xgb", "bayesian"]))
@click.option("-e", "--evaluate",     is_flag=True)
@click.option("-b", "--backtest",     is_flag=True)
def main(model, dataset, algo, evaluate, backtest):
    datasets = ["pregame", "q1", "q1_q2", "q1_q2_q3", "q1_m", "q1_q2_m", "q1_q2_q3_m", "q1.o", "q2.o", "q3.o"] if dataset == "all" else [dataset]
    models = ["total", "winner", "spread", "homepoints", "awaypoints"] if model == "all" else [model]
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for ds in datasets:
            for m in models:
                futures.append(executor.submit(
                    globals()[f"{m}_model"],
                    ds, evaluate, algo, backtest
                ))
        for future in concurrent.futures.as_completed(futures):
            future.result()

def total_model(dataset, ev, algo, backtest):
    run_regression_model(
        "total", lambda d: d["HFINAL"] + d["AFINAL"],
        450, dataset, algo, backtest, ev
    )

def spread_model(dataset, ev, algo, backtest):
    run_regression_model(
        "spread", lambda d: d["HFINAL"] - d["AFINAL"],
        8000, dataset, algo, backtest, ev
    )

def winner_model(dataset, ev, algo, backtest):
    run_classification_model(
        "winner", lambda d: (d["HFINAL"] > d["AFINAL"]).astype(int),
        8000, dataset, algo, backtest, ev
    )

def homepoints_model(dataset, ev, algo, backtest):
    run_regression_model(
        "homepoints", lambda d: d["HFINAL"],
        8000, dataset, algo, backtest, ev
    )

def awaypoints_model(dataset, ev, algo, backtest):
    run_regression_model(
        "awaypoints", lambda d: d["AFINAL"],
        8000, dataset, algo, backtest, ev
    )

def load_data(dataset, start_idx, backtest, model_name):
    if 'o' in dataset:
        data = pd.read_parquet(f"datasets/{dataset.split('.')[0]}/{model_name}.parquet")
    else:
        data = pd.read_parquet(f"datasets/{dataset}.parquet")
    if backtest:
        data = data.iloc[start_idx:BACKTEST_CUTOFF]
    else:
        data = data.iloc[start_idx:]
    return data

def preprocess_features(X, algo):
    X = X.sort_index(axis=1)
    if algo == "xgb":
        X["HOME"] = X["HOME"].astype("category")
        X["AWAY"] = X["AWAY"].astype("category")
    else:
        X = pd.get_dummies(X, columns=['HOME', 'AWAY'], drop_first=True)
    return X

def calculate_weights(X_train, lambda_param=0.01):
    years = X_train["GAME_ID"].apply(lambda x: int(x[2:5]))
    max_year = years.max()
    return np.exp(-lambda_param * (max_year - years))

def run_regression_model(model_name, y_func, start_idx, dataset, algo, backtest, ev):
    data = load_data(dataset, start_idx, backtest, model_name)
    X = data.drop(["HFINAL", "AFINAL"], axis=1)
    X = preprocess_features(X, algo)
    y = y_func(data)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if algo == "linear":
        model = LinearRegression()
        model.fit(X_train.drop("GAME_ID", axis=1), y_train)
    elif algo == "xgb":
        model = xgb.XGBRegressor(
            max_depth=4, learning_rate=0.01, n_estimators=500,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            objective="reg:squarederror", enable_categorical=True
        )
        weights = calculate_weights(X_train)
        model.fit(X_train.drop("GAME_ID", axis=1), y_train, sample_weight=weights)
    elif algo == "bayesian":

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.drop(columns=["GAME_ID"]))
        X_test_scaled = scaler.transform(X_test.drop(columns=["GAME_ID"]))

        with pm.Model() as model:
            # Training data
            X_data = pm.Data("X", X_train_scaled)
            y_data = pm.Data("y", y_train.values)

            intercept = pm.Normal("Intercept", mu=0, sigma=10)
            betas = pm.Normal("Betas", mu=0, sigma=1, shape=X_train_scaled.shape[1])
            sigma = pm.HalfNormal("Sigma", sigma=10)

            mu = intercept + pm.math.dot(X_data, betas)

            # Observed for training only
            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_data)

            trace = pm.sample(500, tune=500, target_accept=0.9, cores=1, return_inferencedata=True)

            # Set test data
            pm.set_data({"X": X_test_scaled})

            # Use the already-registered X_data
            y_pred = pm.Normal("y_pred", mu=intercept + pm.math.dot(X_data, betas), sigma=sigma)
            ppc = pm.sample_posterior_predictive(trace, var_names=["y_pred"], return_inferencedata=True)


        # Posterior predictive mean
        y_pred_mean = ppc.posterior_predictive["y_pred"].mean(dim=["chain", "draw"]).values
        r2 = 1 - np.sum((y_test.values - y_pred_mean) ** 2) / np.sum((y_test.values - np.mean(y_test.values)) ** 2)

        create_output_dirs(model_name, dataset, algo)

        if ev:
            plt.figure(figsize=(10, 5))
            sns.kdeplot(y_test.values, label="Actual", color="black")
            sns.kdeplot(y_pred_mean, label="Predicted", color="blue")
            plt.title(f"{model_name.capitalize()} Posterior Predictive Check")
            plt.legend()
            plt.savefig(f"outputs/{algo}/{dataset}/{model_name}/ppc_kde.png")
            plt.close()

            with open(f"outputs/{algo}/{dataset}/{model_name}/results.log", "w") as f:
                f.write(f"{model_name.upper()} (Bayesian)\n")
                f.write(f"Posterior Predictive R²: {r2:.3f}\n")
                mae = np.mean(np.abs(y_test.values - y_pred_mean))
                rmse = np.sqrt(np.mean((y_test.values - y_pred_mean) ** 2))
                eval_metrics(y_test, y_pred_mean, f)
                f.write(f"MAE: {mae:.3f}\n")
                f.write(f"RMSE: {rmse:.3f}\n")

            # Add standard residuals and actual vs predicted plot for consistency
            residuals(y_test, y_pred_mean, model_name, dataset, algo)
            actual_vs_predicted(y_test, y_pred_mean, model_name, dataset, algo)
            
            

        
    if algo != "bayesian":
        if ev:
            y_pred = model.predict(X_test.drop("GAME_ID", axis=1))
            create_output_dirs(model_name, dataset, algo)
            with open(f"outputs/{algo}/{dataset}/{model_name}/results.log", "w") as f:
                eval_metrics(y_test, y_pred, f)
            residuals(y_test, y_pred, model_name, dataset, algo)
            actual_vs_predicted(y_test, y_pred, model_name, dataset, algo)
            if algo == "xgb":
                feature_importances(
                    model.feature_importances_,
                    X.drop("GAME_ID", axis=1).columns,
                    model_name, dataset, algo
                )
        else:
            create_output_dirs(model_name, dataset, algo)
            y_pred = model.predict(X_test.drop("GAME_ID", axis=1))
            residual = y_pred - y_test
            std = np.std(residual)
            n = len(y_test)
            k = X_test.shape[1]
            se = np.sqrt(np.sum(residual**2) / (n - k))

            residual_path = f'stds/{algo}/{dataset}/{model_name}_residuals.json'
            std_path = f'stds/{algo}/{dataset}/{model_name}_std.json'

            with open(residual_path, 'w') as f:
                json.dump(residual.tolist(), f)
            with open(std_path, 'w') as f:
                json.dump({'std': std, 'se': se}, f)

            if algo == "xgb":
                model.save_model(f"models/{dataset}/{model_name}.ubj")
            else:
                dump(model, f'models/{dataset}/{model_name}.joblib')


def run_classification_model(model_name, y_func, start_idx, dataset, algo, backtest, ev):
    if dataset == "pregame" and model_name in ["winner2", "leader_winner"]:
        return
    
    data = load_data(dataset, start_idx, backtest, model_name)
    X = data.drop(["HFINAL", "AFINAL"], axis=1)
    X = preprocess_features(X, algo)
    y = y_func(data)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBClassifier(
        max_depth=4, learning_rate=0.01, n_estimators=300,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        enable_categorical=True
    )
    weights = calculate_weights(X_train)
    model.fit(X_train.drop("GAME_ID", axis=1), y_train, sample_weight=weights)
    
    if ev:
        y_pred = model.predict(X_test.drop("GAME_ID", axis=1))
        y_proba = model.predict_proba(X_test.drop("GAME_ID", axis=1))[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        
        create_output_dirs(model_name, dataset, algo)
        with open(f"outputs/{algo}/{dataset}/{model_name}/results.log", "w") as f:
            f.write(f"{model_name.upper()}======")
            f.write(f"Accuracy: {accuracy:.4f}\nConfusion Matrix:\n{conf_matrix}\nReport:\n{class_report}")
            
            feature_importances(
                model.feature_importances_,
                X.drop("GAME_ID", axis=1).columns,
                model_name, dataset, algo
            )
        
        plt.figure(figsize=(10, 6))
        plt.hist(y_proba, bins=30, color="skyblue")
        plt.title("Predicted Probabilities Distribution")
        plt.savefig(f"outputs/{algo}/{dataset}/{model_name}/prob_dist.png")
        plt.close()
        
        plt.scatter(y_test, y_proba, alpha=0.5)
        plt.title("Actual vs Predicted Probabilities")
        plt.savefig(f"outputs/{algo}/{dataset}/{model_name}/actual_vs_pred.png")
        plt.close()
    
    else:
        create_output_dirs(model_name, dataset, algo)
        model.save_model(f"models/{dataset}/{model_name}.ubj")

if __name__ == "__main__":
    main()