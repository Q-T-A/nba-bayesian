from nba_api.live.nba.endpoints import boxscore
from nba_api.stats.endpoints import playbyplayv2
import time
import pandas as pd
from tqdm import tqdm

def update_row(row):

    try:
        pbp = playbyplayv2.PlayByPlayV2(row["GAME_ID"]).get_data_frames()[0]
    except:
        print("FAILURE", row["GAME_ID"])
        try:
            time.sleep(5.0)
            pbp = playbyplayv2.PlayByPlayV2(row["GAME_ID"]).get_data_frames()[0]
        except:
            print("FAILURE 2")
            try:
                time.sleep(3.0)
                pbp = playbyplayv2.PlayByPlayV2(row["GAME_ID"]).get_data_frames()[0]
            except:
                return row
                print("FAILURE 3! Exiting...")
                exit()
    time.sleep(0.2)

    pbp.to_parquet(f"pbps/{row['GAME_ID']}.parquet")

tqdm.pandas()

data = pd.read_parquet("datasets/results.parquet")

data = data.progress_apply(update_row, axis=1)