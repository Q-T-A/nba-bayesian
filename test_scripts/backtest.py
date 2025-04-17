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
def compute_confidence_floors(pred, residuals, thresholds, n_bootstrap=1000):
    """
    Calculate the confidence floors for given thresholds using bootstrapping.

    Args:
        pred (float): Predicted value.
        residuals (list): List of residuals to sample from.
        thresholds (list): List of confidence thresholds (e.g., [0.6, 0.65, 0.7]).
        n_bootstrap (int): Number of bootstrap iterations.

    Returns:
        dict: Confidence floor probabilities for each threshold.
    """
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
    type=click.Choice(["xgb", "linear"], case_sensitive=False),
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
    default="winner",
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

    model_names = ["awaypoints", "homepoints", "spread", "total"]

    # Load test data
    test = pd.read_parquet('datasets/odds.parquet')
    test = test.dropna()
    test = test[-750:]

    # Map dataset configurations
    if dataset == "q":
        dname = {1: "q1", 2: "q2", 3: "q3"}
    elif dataset == "q_prev":
        dname = {1: "q1", 2: "q1_q2", 3: "q1_q2_q3"}
    elif dataset == "m":
        dname = {1: "q1_m", 2: "q1_q2_m", 3: "q1_q2_q3_m"}
    else:
        dname = { 1: "pregame", 2: "pregame", 3: "pregame" }
    
    # Load models
    linear_models = [
        {idx: get_model(f"models/{dname[idx]}/{mname}.joblib", "linear") for idx in range(1, 4)}
        for mname in model_names
    ]

    xgb_models = [
        {idx: get_model(f"models/{dname[idx]}/{mname}.ubj", "xgb") for idx in range(1,4)}
        for mname in model_names
    ]

    winmodel = {
            idx: create_classifier(f"models/{dname[idx]}/winner.ubj") for idx in range(1,4)
            }

    cumulative_winnings = {}
    winnings = 0
    plus = 0
    minus = 0
    plus_wins = 0
    minus_wins = 0

    # Process each row in the test data
    for idx, row in tqdm(test.iterrows(), total=len(test), desc="Processing Games"):
        game_id = row["GAME_ID"]
        # Load residuals for confidence calculations
        residuals = {}
        for residual_name in ["total", "spread", "home", "away"]:
            with open(f"stds/{algo}/{dname[1]}/{residual_name}_residuals.json") as f:
                residuals[residual_name] = json.load(f)

        # Iterate through each period and make predictions
        for period, period_name in dname.items():
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
            winner = winmodel[period].predict(X_game)
            winner_probs = winmodel[period].predict_proba(X_game)

            # Add one-hot encoding for HOME and AWAY teams
            if algo == 'linear':
                for i in range(1, 30):
                    X_game[f"HOME_{i}"] = (X["HOME"] == i).astype(int)
                for j in range(1, 30):
                    X_game[f"AWAY_{j}"] = (X["AWAY"] == j).astype(int)
                X_game = X_game.drop(["HOME", "AWAY"], axis=1)
                predictions = [
                    (model_name, model[period].predict(X_game)[0])
                    for model_name, model in zip(model_names, linear_models)
                ]
            elif algo == "xgb":
                X_game["HOME"] = X_game["HOME"].astype("category")
                X_game["AWAY"] = X_game["AWAY"].astype("category")
                predictions = [
                    (name, m[period].predict(X_game)[0])
                    for name, m in zip(model_names, xgb_models)
                ]
            # Process predictions and compute confidence floors
            if kind in ['all', 'points']:
                for pred_name, pred_value in predictions:
                    '''if pred_name in ['away','home']:
                        continue'''
                    residual_key = "home" if pred_name == "homepoints" else \
                                "away" if pred_name == "awaypoints" else pred_name
                    if residual_key == 'spread':
                        continue
                    confidence_floors = compute_confidence_floors(
                        pred_value, residuals[residual_key], [0.56], 100
                    )
                    threshold_key = "56%"
                    # For middle strategy 
                    max_key = "60%"
                    ceiling = confidence_floors[threshold_key]["ceiling"]
                    floor = confidence_floors[threshold_key]["floor"]
                    #upper_ceiling = confidence_floors[max_key]["ceiling"]
                    #lower_floor = confidence_floors[max_key]["floor"]
                    odds_categories = {'total': 'OVERUNDER_', 'homepoints': 'HOVERUNDER_', 'awaypoints': 'AOVERUNDER_'}
                    game_categories = {'total': total, 'homepoints': hpts, 'awaypoints': apts}
                    pnames = {1:'Q1', 2:'Q2', 3:'Q3'}
                    if dataset == "pregame":
                        pnames = { 2:'P' }
                    item = odds_categories[pred_name] + pnames[period]
                    # Bet the under 
                    nobet = False
                    if row[item] > ceiling:
                        total_bets += 1
                        # Line greater than the total outcome 
                        if row[item] > game_categories[pred_name]:
                            total_success += 1
                            winnings += .8696
                        else:
                            winnings -= 1
                    # Bet the over 
                    elif row[item] < floor:
                        total_bets += 1
                        # line was lower than the final 
                        if row[item] < game_categories[pred_name]:
                            total_success += 1
                            winnings += .8696
                        else:
                            winnings -= 1
                    else:
                        nobet = True
                    if not nobet:
                        cumulative_winnings[total_bets]=winnings 
            if kind in ['winner', 'all']:
                for prob in winner_probs:
                    nobet = False
                    probability = prob[1]
                    pnames = {1: 'Q1', 2: 'Q2', 3: 'Q3'}
                    if dataset == "pregame":
                        pnames = { 2: 'Q2' }
                    item = 'HH2H_' + pnames[period]
                    away = 'AH2H_' + pnames[period]
                    away_money  = float(row[away])
                    moneyline = float(row[item])
                    # Correct implied probability calculation
                    if moneyline < 0:
                        implied_prob = abs(moneyline / (moneyline - 100))
                    else:
                        implied_prob = 100 / (moneyline + 100)

                    # Thresholds for betting
                    bet_threshold = 0.1

                    # Bet on the home team
                    if probability - implied_prob >= bet_threshold and probability > .3:
                        print(f'betting at home team at: {implied_prob} ')
                        total_bets += 1
                        if moneyline > 0:
                                plus +=1
                        else: 
                                minus +=1
                        if hpts > apts:  # Home team wins
                            total_success +=1
                            cur = (100/abs(moneyline)) if moneyline < 0 else (moneyline/100)
                            winnings += cur
                            print(f'won: {cur}')
                            if moneyline > 0:
                                plus_wins +=1
                            else: 
                                minus_wins +=1   
                        else:  # Home team loses
                            winnings -= 1
                            print(f'lost: $1')

                    # Bet on the away team
                    elif probability - implied_prob <= -bet_threshold:
                        print(f'betting away team at: {1-implied_prob} ')
                        total_bets += 1
                        if away_money > 0:
                                plus +=1
                        else: 
                                minus +=1
                        if apts > hpts:  # Away team wins
                            total_success += 1
                            cur = ((100 / abs(away_money))) if away_money < 0 else ((away_money / 100))
                            winnings += cur
                            print(f'won: {cur}')
                            if away_money > 0:
                                plus_wins +=1
                            else: 
                                minus_wins +=1   
                        else:  # Away team loses
                            winnings -= 1
                            print(f'lost: $1')


                    else:
                        nobet = True
                    # Update cumulative winnings
                    if not nobet:
                        cumulative_winnings[total_bets] = winnings

            # Default values if no bets are placed
            if total_bets == 0:
                total_bets += 1

            bet = list(cumulative_winnings.keys())
            wins = list(cumulative_winnings.values())
        print(f'Plus bets: {plus}')
        print(f'Minus bets: {minus}')
        try:
            print(f'Plus win rate: {plus_wins/plus}')
            print(f'Minus win rate: {minus_wins/minus}')
        except:
            print('No bets yet')
        print(f'Cumulative win rate: {total_success/total_bets}')
        print(f'Bets placed: {total_bets}, bets won; {total_success}')
        plt.figure(figsize=(10, 6))
        plt.plot(bet, wins, label="Cumulative Winnings", color="blue", linewidth=2)
        plt.xlabel("Number of Bets")
        plt.ylabel("Winnings ($)")
        plt.title("Cumulative Winnings Over Bets")
        plt.legend()
        plt.grid(True)
        
        # Save the plot as an image file
        output_file = "cumulative_winnings.png"
        plt.savefig(output_file, dpi=300)  # Save with 300 dpi for high quality
        plt.close()  # Close the figure to free up memory
        
if __name__ == "__main__":
    main()


