import pandas as pd
from datetime import date
from nba_api.stats.endpoints.leaguegamelog import LeagueGameLog
from tqdm import tqdm
import time
from sys import argv
tqdm.pandas()

ratings = {}

for season in range(2013, 2025):
    req = LeagueGameLog(season=str(season))
    time.sleep(0.6)

    for idx, row in req.get_data_frames()[0].iterrows():
        ratings[row["GAME_ID"]] = row["GAME_DATE"]


def apply_dates(row):
    row["DATE"] = ratings[row["GAME_ID"]]

    return row

results_df = pd.read_parquet("datasets/results.parquet")
results_df = results_df.progress_apply(apply_dates, axis=1)
results_df.to_parquet("datasets/results.parquet")
