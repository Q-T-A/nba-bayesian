import pandas as pd
import xgboost as xgb
from compose import quarter_m_features, quarter_prev_features, optimal_features
import matplotlib.pyplot as plt
import numpy as np
import json
from collections import defaultdict
import click

BACKTEST_CUTOFF = 12500

@click.command()
@click.option(
    "-p",
    "--prop",
    type=click.Choice(
        ["total", "spread", "winner", "homepoints", "awaypoints"], default="total", case_sensitive=False
    ),
)
@click.option(
    "-t",
    "--threshold",
    default = 4.5
)
@click.option(
    "-c",
    "--cutoff",
    default = 10
)
@click.option(
    "-o", 
    "--over", is_flag=True) # just going to make under the default
@click.option(
    "-m",
    "--model",
    type=click.Choice(
        ["prev", "m", "o"], default="prev", case_sensitive=False
    )
)
def main(prop, cutoff, threshold, over, model):
    all_data = pd.read_parquet("datasets/live.parquet")

    all_data["spread"] = all_data["HFINAL"] - all_data["AFINAL"]
    all_data["total"] = all_data["HFINAL"] + all_data["AFINAL"]
    all_data["homepoints"] = all_data["HFINAL"]
    all_data["awaypoints"] = all_data["AFINAL"]
    all_data["winner"] = (all_data["HFINAL"] > all_data["AFINAL"]).astype(int)
    all_data["HOME"] = all_data["HOME"].astype("category")
    all_data["AWAY"] = all_data["AWAY"].astype("category")

    all_data = all_data[BACKTEST_CUTOFF:]

    odds_df = pd.read_parquet("datasets/odds.parquet")
    all_data = all_data.merge(odds_df, on="GAME_ID", how="inner")

    if model == "prev":
        dname = {1: "q1_m", 2: "q1_q2_m", 3: "q1_q2_q3_m"}
    elif model == "m":
        dname = {1: "q1", 2: "q1_q2", 3: "q1_q2_q3"}
    elif model == "o":
        dname = {1: "q1.o", 2: "q2.o", 3: "q3.o"}

    model_names = ["awaypoints", "homepoints", "spread", "total"]

    xgb_models = {
        mname: { 
            idx: create_regressor(f"models/{dname[idx]}/{mname}.ubj") 
            for idx in range(1,4)
        }
        for mname in model_names
    }

    winmodel = {
        idx: create_classifier(f"models/{dname[idx]}/winner.ubj") 
        for idx in range(1,4)
    }

    residuals = defaultdict(dict)

    for mname in model_names:
        for i in range(1, 4):
            with open(f"stds/xgb/{dname[i]}/{mname}_residuals.json") as f:
                residuals[mname][i] = compute_confidence_floors(
                    json.load(f), 0.56, 1000
                )

    if model == "prev":
        X = all_data[quarter_prev_features(2)].drop(["GAME_ID", "HFINAL", "AFINAL"], axis=1).sort_index(axis=1)
    elif model == "m":
        X = all_data[quarter_m_features(2)].drop(["GAME_ID", "HFINAL", "AFINAL"], axis=1).sort_index(axis=1)
    else:
        X = all_data[optimal_features(2, prop)].drop(["GAME_ID", "HFINAL", "AFINAL"], axis=1).sort_index(axis=1)

    for mname in model_names:
        all_data[f"{mname}_pred"] = xgb_models[mname][2].predict(X)
        all_data[f"{mname}_lower"] = all_data[f"{mname}_pred"] + residuals[mname][2][0] # for when function is defined
        all_data[f"{mname}_upper"] = all_data[f"{mname}_pred"] + residuals[mname][2][1]

    all_data["winnerprob"] = winmodel[2].predict_proba(X)[:, 1]

    winnings = pd.Series(np.where(
        ((all_data["total_pred"] + 4.5 < all_data["OVERUNDER_Q2"]) 
         & (all_data["OVERUNDER_Q2"] - all_data["total_pred"] < 10)),
        np.where(
            (all_data["total"] < all_data["OVERUNDER_Q2"]),
            winnings_on_bet(all_data["UNDER_PRICE_Q2"]), 
            -1
        ),
        np.nan
    ), index=all_data.index).dropna()

    winnings.cumsum().plot(kind="line", title="away")
    plt.show()
    plt.close()

    print("All:", len(all_data))
    print("Taken:", winnings.count())
    print("Win pct:", (winnings > 0).mean() * 100)
    print("Total winnings:", winnings.sum())
    print("Max drawdown:", (winnings.cumsum().cummax() - winnings.cumsum()).max())

def winnings_on_bet(odds):
    return np.where(
        odds > 0,
        odds / 100,
        np.where(
            odds < 0,
            100 / np.abs(odds),
            0
        )
    )

def compute_confidence_floors(residuals, threshold, n_bootstrap=1000):
    bootstrap_samples = np.random.choice(residuals, size=(n_bootstrap, len(residuals)), replace=True)
    # maybe look into percentile calc ways idk https://numpy.org/doc/2.1/reference/generated/numpy.percentile.html
    lower_percentile = np.percentile(bootstrap_samples, (1 - threshold) * 100, axis=0)
    upper_percentile = np.percentile(bootstrap_samples, threshold * 100, axis=0)

    return np.min(lower_percentile), np.max(upper_percentile)

def create_classifier(modelname):
    winmodel = xgb.XGBClassifier()
    winmodel.load_model(modelname)
    return winmodel

def create_regressor(name):
    model = xgb.XGBRegressor()
    model.load_model(name)
    return model

if __name__ == "__main__":
    main()