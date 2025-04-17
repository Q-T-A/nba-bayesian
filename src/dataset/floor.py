import pandas as pd
import json
import numpy as np
import random
import click
from live_model import live_model

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
            "floor": round(np.min(lower_percentile), 2),
            "ceiling": round(np.max(upper_percentile), 2)
        }
    return floors

def print_confidence_floors(name, floors):
    """
    Display the confidence floors in a clean format.

    Args:
        name (str): The name of the metric (e.g., Total Points).
        floors (dict): The confidence floors with thresholds.
    """
    print(f"  {name}:")
    for threshold, values in floors.items():
        floor = round(float(values['floor']), 2)
        ceiling = round(float(values['ceiling']), 2)
        print(f"    {threshold}: Floor = {floor}, Ceiling = {ceiling}")


def floor(algo, dataset):
    # Map dataset configurations
    if dataset == "q":
        dname = {
            1: "q1",
            2: "q2",
            3: "q3"
        }
    elif dataset == "q_prev":
        dname = {
            1: "q1",
            2: "q1_q2",
            3: "q1_q2_q3"
        }
    else:
        dname = {
            1: "q1_m",
            2: "q1_q2_m",
            3: "q1_q2_q3_m"
        }
    # Load prediction data
    data_df = live_model(algo, dataset)

    # Confidence thresholds
    thresholds = [.56, 0.6, 0.65, 0.7]

    games = dict()

    # Display predictions and confidence floors
    for _, row in data_df.iterrows():
        game = dict()
        period = row["PERIOD"]
        game["PERIOD"] = period
        game["GAME_ID"] = row["GAME_ID"]
        residuals = {}
        for name in ["total", "spread", "home", "away"]:
            with open(f"stds/{algo}/{dname[period]}/{name}_residuals.json") as f:
                residuals[name] = json.load(f)  # Loading residuals from JSON file

        game["ID"] = row["ID"]
        game["winner"] = row["winner"]
        game["winnerprob"] = round(100 * row['winner_prob'], 2)
        game["total"] = round(row['total'], 2)
        game["spread"] = round(row["spread"], 2)
        game["homepoints"] = round(row["homepoints"], 2)
        game["awaypoints"] = round(row["awaypoints"], 2)

        game["total_conf"] = compute_confidence_floors(row['total'], residuals['total'], thresholds)
        game["spread_conf"] = compute_confidence_floors(row['spread'], residuals['spread'], thresholds)
        game["home_conf"] = compute_confidence_floors(row['homepoints'], residuals['home'], thresholds)
        game["away_conf"] = compute_confidence_floors(row['awaypoints'], residuals['away'], thresholds)

        games[row["GAME_ID"]] = game
    
    return games
        

@click.command()
@click.option(
    "-a",
    "--algo",
    default="xgb",
    type=click.Choice([
        "xgb", "linear"], case_sensitive=False),
)
@click.option(
    "-d",
    "--dataset",
    default="m",
    type=click.Choice([
        "q", "q_prev", "m"], case_sensitive=False),
)
def main(algo, dataset):
    data = floor(algo, dataset)

    for game in data.values():
        print(f"Game: {game['ID']}")
        print(f"  Winner: {game['winner']}")
        print(f"  Winner Probability: {game["winnerprob"]}%")
        print(f"  Total Points: {game['total']}")
        print(f"  Spread Points: {game['spread']}")
        print(f"  Home Points: {game['homepoints']}")
        print(f"  Away Points: {game['awaypoints']}")

        print("Confidence Floors:")
        print_confidence_floors("Total Points", game['total_conf'])
        print_confidence_floors("Spread Points", game['spread_conf'])
        print_confidence_floors("Home Points", game['home_conf'])
        print_confidence_floors("Away Points", game['away_conf'])
        print("\n============================\n")


if __name__ == "__main__":
    main()
