import pandas as pd
from player_pbp_parse import player_pbp_data
from collections import defaultdict, deque
import statistics
from tqdm import tqdm
import json
import ubjson

def player_metrics(live_df, pbps):
    WINDOW_SIZE = 30
    stateful_metrics = defaultdict(lambda : {
        "PIE": deque([10], maxlen=WINDOW_SIZE),
        "TEAM": -1,
    })

    def get_player_metrics(row):
        nonlocal stateful_metrics

        stats = player_pbp_data(pbps[row["GAME_ID"]])

        HPIEs = sorted([statistics.mean(stateful_metrics[player]["PIE"]) for player in stats if stats[player]["team"] == True], reverse=True)
        APIEs = sorted([statistics.mean(stateful_metrics[player]["PIE"]) for player in stats if stats[player]["team"] == False], reverse=True)

        for x in range(5):
            row[f"HPIE_{x + 1}"] = HPIEs[x]
            row[f"APIE_{x + 1}"] = APIEs[x]

        curPIEs = {x: [[] for _ in range(4)] for x in ("H", "A")}

        for p in stats:
            team = "H" if stats[p]["team"] else "A"

            if team == "H" and stateful_metrics[p]["TEAM"] != row["HOME"]:
                stateful_metrics[p]["TEAM"] = row["HOME"]
                stateful_metrics[p]["PIE"] = deque(list(stateful_metrics[p]["PIE"])[-2:], maxlen=WINDOW_SIZE)
            elif team == "A" and stateful_metrics[p]["TEAM"] != row["AWAY"]:
                stateful_metrics[p]["TEAM"] = row["AWAY"]
                stateful_metrics[p]["PIE"] = deque(list(stateful_metrics[p]["PIE"])[-2:], maxlen=WINDOW_SIZE)

            stateful_metrics[p]["PIE"].append((
                (stats[p]["PTS_Q4"] + stats[p]["FGM_Q4"] + stats[p]["TPM_Q4"] + stats[p]["FTM_Q4"] 
                    - stats[p]["FGA_Q4"] - stats[p]["TPA_Q4"] - stats[p]["FTA_Q4"] 
                    + stats[p]["DR_Q4"] + (stats[p]["OR_Q4"] / 2) + stats[p]["AS_Q4"] + stats[p]["ST_Q4"]
                    + (stats[p]["BLK_Q4"] / 2) - stats[p]["FO_Q4"] - stats[p]["TO_Q4"]
                ) / 
                (row[f"{team}PTS_Q4"] + row[f"{team}FGM_Q4"] + row[f"{team}TPM_Q4"] + row[f"{team}FTM_Q4"]
                    - row[f"{team}FGA_Q4"] - row[f"{team}TPA_Q4"] - row[f"{team}FTA_Q4"]
                    + row[f"{team}DR_Q4"] + (row[f"{team}OR_Q4"] / 2) + row[f"{team}AS_Q4"] + row[f"{team}ST_Q4"]
                    + (row[f"{team}BLK_Q4"] / 2) - row[f"{team}FO_Q4"] - row[f"{team}TO_Q4"] 
                )) * 100
            )

            for q in range(1, 4):
                curPIEs[team][q].append(((
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
                

        for q in range(1, 4):
            curPIEs["H"][q].sort(reverse=True)
            curPIEs["A"][q].sort(reverse=True)

            for i in range(5):
                row[f"HPIE_{i + 1}_Q{q}"] = curPIEs["H"][q][i]
                row[f"APIE_{i + 1}_Q{q}"] = curPIEs["A"][q][i]
        
        return row

    print("Player Metrics")
    live_df = live_df.progress_apply(get_player_metrics, axis=1)

    player_averages = {
        str(player): statistics.mean(stateful_metrics[player]["PIE"])
        for player in stateful_metrics
    }

    with open("stds/pie.ubj", "wb") as file:
        file.write(ubjson.dumpb(player_averages))

    return live_df

if __name__ == "__main__":
    tqdm.pandas()
    live_df = pd.read_parquet("datasets/live.parquet")

    pbps = {}
    def set_pbps(row):
        pbps[row['GAME_ID']] = pd.read_parquet(f"pbps/{row['GAME_ID']}.parquet")

    print("Set pbps")
    live_df.progress_apply(set_pbps, axis=1)

    live_df = player_metrics(live_df, pbps)

    live_df.to_parquet("datasets/live.parquet")