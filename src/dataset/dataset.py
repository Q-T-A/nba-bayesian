from nba_api.stats.endpoints import playbyplayv2
from nba_api.stats.endpoints import leaguegamelog
from parse_pbp import get_pbp_data
import pandas as pd
import time
from datetime import datetime
import json
from tqdm import tqdm
from collections import defaultdict, deque
import numpy as np
from sklearn import linear_model
from metrics import update_metrics
import get_odds
from player_metrics import player_metrics
import compose
import concurrent.futures
from multiprocessing import Manager


nba_team_abbreviations = [
    "ATL",
    "BOS",
    "BKN",
    "CHA",
    "CHI",
    "CLE",
    "DAL",
    "DEN",
    "DET",
    "GSW",
    "HOU",
    "IND",
    "LAC",
    "LAL",
    "MEM",
    "MIA",
    "MIL",
    "MIN",
    "NOP",
    "NYK",
    "OKC",
    "ORL",
    "PHI",
    "PHX",
    "POR",
    "SAC",
    "SAS",
    "TOR",
    "UTA",
    "WAS",
]

abbrev_map = {abbrev: idx for idx, abbrev in enumerate(nba_team_abbreviations)}
team_map = {idx: abbrev for idx, abbrev in enumerate(nba_team_abbreviations)}


def update_game_results():
    def transform_row(row):
        # if winner is away
        if row[("MATCHUP", "W")].split()[1] == "@":
            home = "L"
            away = "W"
        else:
            home = "W"
            away = "L"

        row["HOME"] = abbrev_map[row[("TEAM_ABBREVIATION", home)]]
        row["AWAY"] = abbrev_map[row[("TEAM_ABBREVIATION", away)]]
        row["HFINAL"] = row[("PTS", home)]
        row["AFINAL"] = row[("PTS", away)]

        row["DATE"] = row[("GAME_DATE", home)]

        return row

    cur_df = pd.read_parquet("datasets/results.parquet")

    year = datetime.now().year
    gamelog = leaguegamelog.LeagueGameLog(counter=5, season=str(2024))
    gamelog_df = gamelog.get_data_frames()[0]
    gamelog_df = gamelog_df[gamelog_df["WL"].isin(["W", "L"])]

    new_df = gamelog_df.pivot(index="GAME_ID", columns="WL")
    new_df = new_df.apply(transform_row, axis=1)
    new_df = new_df[["HOME", "AWAY", "HFINAL", "AFINAL", "DATE"]].reset_index()
    new_df.columns = new_df.columns.map(lambda x: x[0])

    latest_df = new_df[~new_df["GAME_ID"].isin(cur_df["GAME_ID"])]

    print("Download PBPs")
    for idx in latest_df["GAME_ID"]:
        try:
            pbp = playbyplayv2.PlayByPlayV2(idx)
        except:
            print("ERROR", idx)
            time.sleep(1.5)
            try:
                pbp = playbyplayv2.PlayByPlayV2(idx)
            except:
                print("ERROR 2", idx)
                continue
        df = pbp.get_data_frames()[0]

        df.to_parquet(f"pbps/{idx}.parquet", engine="pyarrow", compression="snappy")
        time.sleep(0.6)

    new_df = pd.concat([cur_df, latest_df], ignore_index=True)

    return new_df, latest_df

def update_elo(results_df):
    def generate_ratings(dataset):
        X = []
        Y = []

        for home, games in dataset.items():
            for away, winner in games:
                v = np.zeros(30)
                v[home] = 1
                v[away] = -1
                
                X.append(v)
                Y.append(winner)
            
        if len(X) > 450:
            clf = linear_model.LogisticRegression()
            clf.fit(X,Y)

            return (clf.coef_[0] * 173 + 1500, clf.intercept_[0])
        else:
            return ([1500 for _ in range(30)], 0)
    
    def add_elo(row):
        nonlocal dataset
        nonlocal ratings
        row["HELO"] = ratings[row["HOME"]]
        row["AELO"] = ratings[row["AWAY"]]

        ratings, _ = generate_ratings(dataset)

        dataset[row["HOME"]].append((row["AWAY"], row["HFINAL"] > row["AFINAL"]))

        return row
    
    def add_elo_quarter(row, quarter):
        nonlocal dataset
        nonlocal ratings
        row[f"HELO_Q{quarter}"] = ratings[row["HOME"]]
        row[f"AELO_Q{quarter}"] = ratings[row["AWAY"]]

        ratings, _ = generate_ratings(dataset)

        dataset[row["HOME"]].append((row["AWAY"], row[f"HPOINTS_Q{quarter}"] > row[f"APOINTS_Q{quarter}"]))

        return row
    
    ratings_exp = {}

    MAX_HOME_GAMES_PER_TEAM = 30
    dataset = defaultdict(lambda: deque(maxlen=MAX_HOME_GAMES_PER_TEAM))
    ratings = [1500 for _ in range(30)]

    results_df = results_df.progress_apply(add_elo, axis=1)
    
    ratings_exp[0] = {idx: ratings[idx] for idx in range(30)}

    for quarter in range(1, 5):
        print(f"Elo Q{quarter}")
        MAX_HOME_GAMES_PER_TEAM = 30
        dataset = defaultdict(lambda: deque(maxlen=MAX_HOME_GAMES_PER_TEAM))
        ratings = [1500 for _ in range(30)]
        results_df = results_df.progress_apply(add_elo_quarter, axis=1, args=(quarter,))
        ratings_exp[quarter] = {idx: ratings[idx] for idx in range(30)}

    with open("stds/ratings.json", "w") as file:
        file.write(json.dumps(ratings_exp, indent=4))

    return results_df

def main():
    tqdm.pandas()

    print("results")
    results_df, _ = update_game_results()
    # results_df = pd.read_parquet("datasets/results.parquet")

    print("Set pbps")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(lambda x: (x, pd.read_parquet(f'pbps/{x}.parquet')), results_df["GAME_ID"])

    pbps = { game_id: df for game_id, df in results }

    print("Get stats from pbp")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        stats_list = list(executor.map(get_pbp_data, pbps.values()))
    
    live_stats_df = pd.DataFrame(stats_list).reset_index(drop=True)

    print("Update Elo")

    results_df["HPOINTS_Q1"] = live_stats_df["HPTS_Q1"]
    results_df["APOINTS_Q1"] = live_stats_df["APTS_Q1"]
    results_df["HPOINTS_Q2"] = live_stats_df["HPTS_Q2"] - live_stats_df["HPTS_Q1"]
    results_df["APOINTS_Q2"] = live_stats_df["APTS_Q2"] - live_stats_df["APTS_Q1"]
    results_df["HPOINTS_Q3"] = live_stats_df["HPTS_Q3"] - live_stats_df["HPTS_Q2"]
    results_df["APOINTS_Q3"] = live_stats_df["APTS_Q3"] - live_stats_df["APTS_Q2"]
    results_df["HPOINTS_Q4"] = results_df["HFINAL"] - live_stats_df["HPTS_Q3"]
    results_df["APOINTS_Q4"] = results_df["AFINAL"] - live_stats_df["APTS_Q3"]

    results_df = update_elo(results_df)

    results_df = results_df.drop([
        "HPOINTS_Q1",
        "APOINTS_Q1",
        "HPOINTS_Q2",
        "APOINTS_Q2",
        "HPOINTS_Q3",
        "APOINTS_Q3",
        "HPOINTS_Q4",
        "APOINTS_Q4"], axis=1)
    
    #odds_df = get_odds.main(results_df)

    live_df = pd.concat([results_df, live_stats_df], axis=1)
    live_df = live_df.copy()
    live_df = update_metrics(live_df)
    live_df = live_df.copy()
    live_df = player_metrics(live_df, pbps)

    #live_df = pd.concat([live_df, odds_df.drop(["GAME_ID"], axis=1)], axis=1)

    results_df.to_parquet("datasets/results.parquet")
    live_df.to_parquet("datasets/live.parquet")
    #odds_df.to_parquet("datasets/odds.parquet")

    compose.main()

if __name__ == "__main__":
    main()
