import pandas as pd
import json
import numpy as np
import click
import xgboost as xgb
from joblib import load
import random
from tqdm import tqdm
import matplotlib 
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

def create_classifier(modelname):
        winmodel = xgb.XGBClassifier()
        winmodel.load_model(modelname)
        return winmodel

def load_bayesian_preds(game_id, period_name, mname,algo):
    path = f"backtest_preds/{algo}_{period_name}_{mname}.json"
    with open(path) as f:
        data = json.load(f)
    idx = data["GAME_ID"].index(game_id)
    return {
        "pred": data["y_pred_mean"][idx],
        "floor": data["lower"][idx],
        "ceiling": data["upper"][idx]
    }

def compute_confidence_floors(pred, residuals, thresholds, n_bootstrap=100000):
    
    # Bootstrap resampling
    bootstrap_samples = []
    for _ in range(n_bootstrap):
        sample_residuals = random.choices(residuals, k=len(residuals))  # Resample with replacement
        bootstrapped_values = pred + np.array(sample_residuals)  # Predicted value + residuals
        bootstrap_samples.append(bootstrapped_values)

    # Compute confidence intervals from bootstrapped samples
    bootstrap_samples = np.array(bootstrap_samples)
    floors = {}
    for threshold in thresholds:
        lower_percentile = np.percentile(bootstrap_samples, (1 - threshold) * 100, axis=0)
        upper_percentile = np.percentile(bootstrap_samples, threshold * 100, axis=0)
        floors[f"{int(threshold * 100)}%"] = {
            "floor": np.min(lower_percentile),
            "ceiling": np.max(upper_percentile)
        }
    return floors

@click.command()
@click.option(
    "-a",
    "--algo",
    default="linear",
    type=click.Choice(["xgb", "linear", "bayesian_t","bayesian_informative"], case_sensitive=False),
)
@click.option(
    "-d",
    "--dataset",
    default="q_prev",
    type=click.Choice(["q", "q_prev", "m", "pregame"], case_sensitive=False),
)
@click.option(
    "-k",
    "--kind",
    default="points",
    type=click.Choice(["winner", "points", "all"], case_sensitive=False),
)
def main(algo, dataset, kind):
    def get_model(name, algo):
        """Load model based on algorithm type."""
        print(f"Loading model: {name}, Algorithm: {algo}")
        if algo == "xgb":
            model = xgb.XGBRegressor()
            model.load_model(name)
        else:
            model = load(name)
        return model

    # Initialize cumulative counters outside the row loop
    total_bets = 0
    total_success = 0

    model_names = ["total"]

    # Load test data
    test = pd.read_parquet('datasets/odds.parquet')
    test = test.dropna()
    test = test[-500:]

    dname = {1: "q1", 2: "q1_q2", 3: "q1_q2_q3"}
    
    

    linear_models = {
        m: get_model(f"models/{dname[2]}/{m}.joblib", "linear")
        for m in model_names
    }

    xgb_models = {
        m: get_model(f"models/{dname[2]}/{m}.ubj",     "xgb")
        for m in model_names
    }

    cumulative_winnings = {}
    winnings = 0

    # Process each row in the test data
    for idx, row in tqdm(test.iterrows(), total=len(test), desc="Processing Games"):
        game_id = row["GAME_ID"]
        if algo in ["xgb", "linear"]:
            residuals = {}
            for residual_name in ["total"]:
                with open(f"stds/{algo}/{dname[2]}/{residual_name}_residuals.json") as f:
                    residuals[residual_name] = json.load(f)

        # Iterate through each period and make predictions
        for period, period_name in dname.items():
            print(period)
            if period in [1,3]:
                continue
            # Load feature data for the current period
            X = pd.read_parquet(f'datasets/{period_name}.parquet')

            X_game = X[X["GAME_ID"] == game_id]
            X_game = X_game.drop(["GAME_ID"], axis=1)

            # Check if X_game is empty
            if X_game.empty:
                print(f"No data found for GAME_ID: {game_id}")
                continue  # Skip this iteration and move to the next game
            total = float(X_game['HFINAL'].iloc[0] + X_game['AFINAL'].iloc[0])
            spread = float(X_game['HFINAL'].iloc[0] - X_game['AFINAL'].iloc[0])
            hpts = float(X_game['HFINAL'].iloc[0])
            apts = float(X_game['AFINAL'].iloc[0])

            X_game = X_game.drop(["HFINAL", "AFINAL"], axis=1)
            X_game = X_game.sort_index(axis=1)

            # Add one-hot encoding for HOME and AWAY teams
            if algo == 'linear':
                X_game = X_game.sort_index(axis=1)
                for i in range(1, 30):
                    X_game[f"HOME_{i}"] = (X["HOME"] == i).astype(int)
                for j in range(1, 30):
                    X_game[f"AWAY_{j}"] = (X["AWAY"] == j).astype(int)
                X_game = X_game.drop(["HOME", "AWAY"], axis=1)
                predictions = [
                    (mname, linear_models[mname].predict(X_game)[0])
                    for mname in model_names
                ]
            elif algo == "xgb":
                X_game = X_game.sort_index(axis=1)
                X_game["HOME"] = X_game["HOME"].astype("category")
                X_game["AWAY"] = X_game["AWAY"].astype("category")
                predictions = [
                    (mname, xgb_models[mname].predict(X_game)[0])
                    for mname in model_names
                ]
            
            elif algo.startswith('bayes'):
                predictions = []
                for mname in model_names:
                    result = load_bayesian_preds(game_id, period_name, mname,algo)
                    predictions.append((mname, result["pred"]))
                    # Override ceiling/floor from JSON instead of bootstrap
                    floor = result["floor"]
                    ceiling = result["ceiling"]

            # Process predictions and compute confidence floors
            if kind in ['all', 'points']:
                for pred_name, pred_value in predictions:
                    if pred_name in ['away','home']:
                        continue
                    residual_key = "home" if pred_name == "homepoints" else \
                                "away" if pred_name == "awaypoints" else pred_name
                    if residual_key == 'spread':
                        continue
                    if algo not in ["bayesian_t", "bayesian_informative"]:
                        confidence_floors = compute_confidence_floors(
                            pred_value, residuals[residual_key], [0.56], 100
                        )
                        threshold_key = "56%"
                        ceiling = confidence_floors[threshold_key]["ceiling"]
                        floor = confidence_floors[threshold_key]["floor"]
                        
                    #upper_ceiling = confidence_floors[max_key]["ceiling"]
                    #lower_floor = confidence_floors[max_key]["floor"]
                    odds_categories = {'total': 'OVERUNDER_', 'homepoints': 'HOVERUNDER_', 'awaypoints': 'AOVERUNDER_'}
                    game_categories = {'total': total, 'homepoints': hpts, 'awaypoints': apts}
                    pnames = {1:'Q1', 2:'Q2', 3:'Q3'}
                    item = odds_categories[pred_name] + pnames[period]
                    # Bet the under 
                    nobet = False
                    print(f"line: {row[item]}")
                    print(f"ceiling: {ceiling}, floor: {floor}")
                    print(f"Final: {game_categories[pred_name]}")
                    if row[item] > ceiling:
                        total_bets += 1
                        # Line greater than the total outcome 
                        if row[item] > game_categories[pred_name]:
                            total_success += 1
                            winnings += .91
                        else:
                            winnings -= 1
                    # Bet the over 
                    elif row[item] < floor:
                        total_bets += 1
                        # line was lower than the final 
                        if row[item] < game_categories[pred_name]:
                            total_success += 1
                            winnings += .91
                        else:
                            winnings -= 1
                    else:
                        nobet = True
                    if not nobet:
                        cumulative_winnings[total_bets]=winnings

            # Default values if no bets are placed
            if total_bets == 0:
                total_bets += 1

            bet = list(cumulative_winnings.keys())
            wins = list(cumulative_winnings.values())
        
    import seaborn as sns
    sns.set_theme(style="darkgrid")

    plt.figure(figsize=(12, 6))
    plt.plot(bet, wins, label="Cumulative Winnings", color="#4cc9f0", linewidth=2.5)

    # Highlight final point
    plt.scatter(bet[-1], wins[-1], color="#f72585", zorder=5)
    plt.text(bet[-1], wins[-1], f"${wins[-1]:.2f}", color="#f72585", fontsize=12, ha='right')

    # Stats box
    plt.annotate(
        f"Win rate: {total_success / total_bets:.1%}\n"
        f"Total bets: {total_bets}\n"
        f"Net: ${wins[-1]:.2f}\n"
        f"ROI: {(wins[-1] / total_bets):.2%}",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", fc="black", ec="white", lw=1, alpha=0.8),
        ha="left", va="top", color="white"
    )

    plt.title(f"{algo.upper()} · Cumulative Winnings Over Bets", fontsize=16, color="white", weight="bold")
    plt.xlabel("Number of Bets", fontsize=12, color="white")
    plt.ylabel("Profit ($)", fontsize=12, color="white")
    plt.xticks(color="white")
    plt.yticks(color="white")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()

    # Save based on algo
    filename = f"cumulative_winnings_{algo}.png"
    plt.savefig(filename, dpi=300, facecolor="#222222")
    plt.close()
    print(f"[✓] Saved plot to {filename}")

        
if __name__ == "__main__":
    main()


