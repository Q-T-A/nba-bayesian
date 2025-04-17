import xgboost as xgb
import numpy as np
import pandas as pd
from get_live_games import get_live_games
from datetime import date
from joblib import load
import os
import json
import ubjson
import click
from compose import quarter_features, quarter_prev_features, quarter_m_features
from datetime import date
from live_player_pbp import live_player_pbp
from nba_api.live.nba.endpoints import playbyplay

def live_model(algo, dataset):
    prev = False
    minutes = False
    if dataset == "q_prev":
        prev = True
    elif dataset == "m":
        prev = True
        minutes = True

    data = get_live_games(prev, minutes)

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

    def create_classifier(modelname):
        winmodel = xgb.XGBClassifier()
        winmodel.load_model(modelname)
        return winmodel

    winmodel = {
        idx: create_classifier(f"models/{dname[idx]}/winner.ubj") for idx in range(1,4)
    }

    winmodel2 = {
        idx: create_classifier(f"models/{dname[idx]}/winner2.ubj") for idx in range(1,4)
    }

    def get_model(name, algo):
        if algo == "xgb":
            model = xgb.XGBRegressor()
            model.load_model(name)
        else:
            model = load(name)
        return model

    model_names = [
        "awaypoints",
        "homepoints",
        "spread",
        "total"
    ]

    linear_models = [
        {idx: get_model(f"models/{dname[idx]}/{mname}.joblib", "linear") for idx in range(1,4)}
        for mname in model_names
    ]
    xgb_models = [
        {idx: get_model(f"models/{dname[idx]}/{mname}.ubj", "xgb") for idx in range(1,4)}
        for mname in model_names
    ]

    with open("stds/ratings.json", "r") as file:
        ratings = json.loads(file.read())
    
    with open("stds/pie.ubj", "rb") as file:
        player_metrics = ubjson.load(file)

    # with open("stds/player_metrics.json", "r") as file:
    #     player_metrics = json.loads(file.read())

    def add_ratings(row):
        nonlocal ratings
        row["HELO"] = ratings['0'][str(int(row["HOME"]))]
        row["AELO"] = ratings['0'][str(int(row["AWAY"]))]

        row["HORATING"] = ratings['ORATINGS'][str(int(row["HOME"]))]
        row["AORATING"] = ratings['ORATINGS'][str(int(row["AWAY"]))]

        row["HDRATING"] = ratings['DRATINGS'][str(int(row["HOME"]))]
        row["ADRATING"] = ratings['DRATINGS'][str(int(row["AWAY"]))]

        row["HTCP"] = ratings['TRUE_SHOOTING'][str(int(row["HOME"]))]
        row["ATCP"] = ratings['TRUE_SHOOTING'][str(int(row["AWAY"]))]

        row["HAPCT"] = ratings['ASSIST_PCT'][str(int(row["HOME"]))]
        row["AAPCT"] = ratings['ASSIST_PCT'][str(int(row["AWAY"]))]

        row["HTOR"] = ratings['TO_RATIO'][str(int(row["HOME"]))]
        row["ATOR"] = ratings['TO_RATIO'][str(int(row["AWAY"]))]

        row["HREST"] = min((date.today() - date(*map(int, ratings["LAST_GAME"][str(int(row["HOME"]))].split("-")))).days - 1, 14)
        row["AREST"] = min((date.today() - date(*map(int, ratings["LAST_GAME"][str(int(row["AWAY"]))].split("-")))).days - 1, 14)

        row["HAVG"] = ratings['AVG_SCORE'][str(int(row["HOME"]))]
        row["AAVG"] = ratings['AVG_SCORE'][str(int(row["AWAY"]))]
        
        row["HPACE_AVG"] = ratings['AVG_PACE'][str(int(row["HOME"]))]
        row["APACE_AVG"] = ratings['AVG_PACE'][str(int(row["AWAY"]))]
        
        row["HPACE_AVG_Q4"] = ratings['AVG_PACE_Q4'][str(int(row["HOME"]))]
        row["APACE_AVG_Q4"] = ratings['AVG_PACE_Q4'][str(int(row["AWAY"]))]

        for i in range(1,5):
            row[f"HELO_Q{i}"] = ratings[str(i)][str(int(row["HOME"]))]
            row[f"AELO_Q{i}"] = ratings[str(i)][str(int(row["AWAY"]))]
        return row
    
    def add_player_metrics(row):
        stats = live_player_pbp(playbyplay.PlayByPlay(row["GAME_ID"]).get_dict(), row["HOME_V"], row["AWAY_V"])
        
        curPIEs = {"H": [], "A": []}
        player_ids = {"H": [], "A": []}

        for p in stats:
            team = "H" if stats[p]["team"] else "A"
            player_ids[team].append(p)

            q = row["period"]
            curPIEs[team].append(((
                stats[p][f"PTS_Q{q}"] + stats[p][f"FGM_Q{q}"] + stats[p][f"TPM_Q{q}"] + stats[p][f"FTM_Q{q}"] 
                - stats[p][f"FGA_Q{q}"] - stats[p][f"TPA_Q{q}"] - stats[p][f"FTA_Q{q}"] 
                + stats[p][f"DR_Q{q}"] + (stats[p][f"OR_Q{q}"] / 2) + stats[p][f"AS_Q{q}"] + stats[p][f"ST_Q{q}"]
                + (stats[p][f"BLK_Q{q}"] / 2) - stats[p][f"FO_Q{q}"] - stats[p][f"TO_Q{q}"]
            ) / (max((
                row[f"{team}PTS_Q{q}"] + row[f"{team}FGM_Q{q}"] + row[f"{team}TPM_Q{q}"] + row[f"{team}FTM_Q{q}"]
                - row[f"{team}FGA_Q{q}"] - row[f"{team}TPA_Q{q}"] - row[f"{team}FTA_Q{q}"]
                + row[f"{team}DR_Q{q}"] + (row[f"{team}OR_Q{q}"] / 2) + row[f"{team}AS_Q{q}"] + row[f"{team}ST_Q{q}"]
                + (row[f"{team}BLK_Q{q}"] / 2) - row[f"{team}FO_Q{q}"] - row[f"{team}TO_Q{q}"]), 1)
            )) * 100)
        
        curPIEs["A"].sort(reverse=True)
        curPIEs["H"].sort(reverse=True)
        for i in range(5):
            row[f"HPIE_{i + 1}_Q{q}"] = curPIEs["H"][i]
            row[f"APIE_{i + 1}_Q{q}"] = curPIEs["A"][i]

        HPIEs = sorted([player_metrics[str(player)] for player in player_ids["H"] if str(player) in player_metrics], reverse=True)
        APIEs = sorted([player_metrics[str(player)] for player in player_ids["A"] if str(player) in player_metrics], reverse=True)

        for x in range(5):
            row[f"HPIE_{x + 1}"] = HPIEs[x]
            row[f"APIE_{x + 1}"] = APIEs[x]

        return row

    res = []
    for df in data:
        period = df.iloc[0]["period"]
        df = df.apply(add_player_metrics, axis=1)
        game_id = df.iloc[0]["GAME_ID"]
        X = df.drop(["HOME_V", "AWAY_V", "GAME_ID", "period"], axis=1)
        X = X.apply(add_ratings, axis=1)

        X["HOME"] = X["HOME"].astype("category")
        X["AWAY"] = X["AWAY"].astype("category")

        X['ODDS'] = 1 / (1 + 10 ** (-(X['HELO'] - X['AELO']) / 400))
        
        if dataset == "q":
            X = X[[f for f in quarter_features(period) if f not in {'GAME_ID', 'HFINAL', 'AFINAL'}]]
        elif dataset == "q_prev":
            #X = X[[f for f in [*quarter_prev_features(period), 'ODDS'] if f not in {'GAME_ID', 'HFINAL', 'AFINAL'}]]
            X = X[[f for f in quarter_prev_features(period) if f not in {'GAME_ID', 'HFINAL', 'AFINAL'}]]
        elif dataset == "m":
            X = X[[f for f in quarter_m_features(period) if f not in {'GAME_ID', 'HFINAL', 'AFINAL'}]]
        X = X.sort_index(axis = 1)

        winner = winmodel[period].predict(X)
        winner_probs = winmodel[period].predict_proba(X)
        
        # winner2 = winmodel2[period].predict(X)
        #winner_probs2 = winmodel2[period].predict_proba(X)
        X = df.drop(["HOME_V", "AWAY_V", "GAME_ID", "period"], axis=1)
        X = X.apply(add_ratings, axis=1)
        X['ODDS'] = 1 / (1 + 10 ** (-(X['HELO'] - X['AELO']) / 400))

        if dataset == "q":
            X = X[[f for f in quarter_features(period) if f not in {'GAME_ID', 'HFINAL', 'AFINAL'}]]
        elif dataset == "q_prev":
            X = X[[f for f in quarter_prev_features(period) if f not in {'GAME_ID', 'HFINAL', 'AFINAL'}]]
        elif dataset == "m":

            X = X[[f for f in quarter_m_features(period) if f not in {'GAME_ID', 'HFINAL', 'AFINAL'}]]

        X["HOME"] = X["HOME"].astype("category")
        X["AWAY"] = X["AWAY"].astype("category")
        
        X = X.sort_index(axis = 1)
        if algo == "linear":
            for i in range(1, 30):
                X[f'HOME_{i}'] = (X['HOME'] == i).astype(int)  # Assign 1 if HOME == i, else 0
            for j in range(1, 30):
                X[f'AWAY_{j}'] = (X['AWAY'] == j).astype(int)  # Assign 1 if AWAY == j, else 0

            X = X.drop(['HOME', 'AWAY'], axis=1)

        if algo == "linear":
            preds = [
                (name, m[period].predict(X)[0])
                for name, m in zip(model_names, linear_models)
            ]
        elif algo == "xgb":
            preds = [
                (name, m[period].predict(X)[0])
                for name, m in zip(model_names, xgb_models)
            ]
        else:
            sep = 0.6
            preds_xgb = [mx[period].predict(X)[0] for mx in xgb_models]

            for i in range(1, 30):
                X[f'HOME_{i}'] = (X['HOME'] == i).astype(int)  # Assign 1 if HOME == i, else 0
            for j in range(1, 30):
                X[f'AWAY_{j}'] = (X['AWAY'] == j).astype(int)  # Assign 1 if AWAY == j, else 0

            X = X.drop(['HOME', 'AWAY'], axis=1)

            preds_linear = [ml[period].predict(X)[0] for ml in linear_models]

            preds = [
                (name, (lp * sep) + (xp * (1 - sep)))
                for name, lp, xp in zip(model_names, preds_linear, preds_xgb)
            ]

        row = df.iloc[0]

        row_pred = {
            "GAME_ID": game_id,
            "ID": f'{row["AWAY_V"]} @ {row["HOME_V"]}',
            "PERIOD": period,
            **{name: res for name, res in preds},
            "winner": row["HOME_V"] if winner[0] else row["AWAY_V"],
            "winner_prob": winner_probs[0][winner[0]],
           # "second_winner": row["HOME_V"] if winner2[0] else row["AWAY_V"],
            #"winner_prob2": winner_probs2[0][winner[0]]
        }
        res.append(row_pred)

    data = pd.DataFrame(res)
    return data

@click.command()
@click.option(
    "-a",
    "--algo",
    default="linear",
    type=click.Choice(
        ["linear", "xgb", "ensemble"], case_sensitive=False
    ),
)
@click.option(
    "-d",
    "--dataset",
    default="m",
    type=click.Choice(
        ["q", "q_prev", "m"], case_sensitive=False
    ),
)
def main(algo, dataset):
    live_model(algo, dataset)

if __name__ == "__main__":
    live_df = main()
    live_df.to_parquet(f"livedata.parquet", engine="pyarrow", compression="snappy")
