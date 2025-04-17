from collections import deque, defaultdict
import pandas as pd
import statistics
import json
from datetime import date
from tqdm import tqdm
import compose
tqdm.pandas()

def update_metrics(live_df: pd.DataFrame) -> pd.DataFrame:

    avg_func = statistics.mean

    MAX_HOME_GAMES_PER_TEAM = 30

    dataset = {
        "PACE_AVG":     defaultdict(lambda: deque([2], maxlen=MAX_HOME_GAMES_PER_TEAM)),
        "PACE_AVG_Q4":  defaultdict(lambda: deque([2], maxlen=MAX_HOME_GAMES_PER_TEAM)),
        "AVG":          defaultdict(lambda: deque([112], maxlen=MAX_HOME_GAMES_PER_TEAM)),
        "ORATING":      defaultdict(lambda: deque([215], maxlen=MAX_HOME_GAMES_PER_TEAM)),
        "DRATING":      defaultdict(lambda: deque([115], maxlen=MAX_HOME_GAMES_PER_TEAM)),
        "TCP":          defaultdict(lambda: deque([57], maxlen=MAX_HOME_GAMES_PER_TEAM)),
        "APCT":         defaultdict(lambda: deque([66], maxlen=MAX_HOME_GAMES_PER_TEAM)),
        "TOR":          defaultdict(lambda: deque([28], maxlen=MAX_HOME_GAMES_PER_TEAM)),
    }

    def all_stats(row):
        nonlocal dataset

        row["HPACE_AVG"]        = avg_func(dataset["PACE_AVG"][row["HOME"]])
        row["APACE_AVG"]        = avg_func(dataset["PACE_AVG"][row["AWAY"]])
        row["HPACE_AVG_Q4"]     = avg_func(dataset["PACE_AVG_Q4"][row["HOME"]])
        row["APACE_AVG_Q4"]     = avg_func(dataset["PACE_AVG_Q4"][row["AWAY"]])
        row["HAVG"]             = avg_func(dataset["AVG"][row["HOME"]])
        row["AAVG"]             = avg_func(dataset["AVG"][row["AWAY"]])
        row["HORATING"]         = avg_func(dataset["ORATING"][row["HOME"]])
        row["AORATING"]         = avg_func(dataset["ORATING"][row["AWAY"]])
        row["HDRATING"]         = avg_func(dataset["DRATING"][row["HOME"]])
        row["ADRATING"]         = avg_func(dataset["DRATING"][row["AWAY"]])
        row["HTCP"]             = avg_func(dataset["TCP"][row["HOME"]])
        row["ATCP"]             = avg_func(dataset["TCP"][row["AWAY"]])
        row["HAPCT"]            = avg_func(dataset["APCT"][row["HOME"]])
        row["AAPCT"]            = avg_func(dataset["APCT"][row["AWAY"]])
        row["HTOR"]             = avg_func(dataset["TOR"][row["HOME"]])
        row["ATOR"]             = avg_func(dataset["TOR"][row["AWAY"]])

        dataset["PACE_AVG"][row["HOME"]].append(
            (row["HFGA_Q4"] + row["HTPA_Q4"] + (0.44 * row["HFTA_Q4"]) - row["HOR_Q4"] + row["HTO_Q4"]) / row["MIN"]
        )
        dataset["PACE_AVG"][row["HOME"]].append(
            (row["AFGA_Q4"] + row["ATPA_Q4"] + (0.44 * row["AFTA_Q4"]) - row["AOR_Q4"] + row["ATO_Q4"]) / row["MIN"]
        )

        dataset["PACE_AVG_Q4"][row["HOME"]].append(
            ((row["HFGA_Q4"] + row["HTPA_Q4"] + (0.44 * row["HFTA_Q4"]) - row["HOR_Q4"] + row["HTO_Q4"])
            - (row["HFGA_Q3"] + row["HTPA_Q3"] + (0.44 * row["HFTA_Q3"]) - row["HOR_Q3"] + row["HTO_Q3"])) / (row["MIN"] - 36)
        )
        dataset["PACE_AVG_Q4"][row["AWAY"]].append(
            ((row["AFGA_Q4"] + row["ATPA_Q4"] + (0.44 * row["AFTA_Q4"]) - row["AOR_Q4"] + row["ATO_Q4"])
            - (row["AFGA_Q3"] + row["ATPA_Q3"] + (0.44 * row["AFTA_Q3"]) - row["AOR_Q3"] + row["ATO_Q3"])) / (row["MIN"] - 36)
        )

        dataset["AVG"][row["HOME"]].append(row["HFINAL"])
        dataset["AVG"][row["AWAY"]].append(row["AFINAL"])

        dataset["ORATING"][row["HOME"]].append(
            100 * (row["HFINAL"] / (
                0.5 * ((row["HFGA_Q4"] + row["HTPA_Q4"] + (0.4 * row["HFTA_Q4"]) - (1.07 * row["HOR_Q4"])) + row["HTO_Q4"])
            ))
        )
        dataset["ORATING"][row["AWAY"]].append(
            100 * (row["AFINAL"] / (
                0.5 * ((row["AFGA_Q4"] + row["ATPA_Q4"] + (0.4 * row["AFTA_Q4"]) - (1.07 * row["AOR_Q4"])) + row["ATO_Q4"])
            ))
        )

        dataset["DRATING"][row["HOME"]].append(
            100 * (row["AFINAL"] / 
            (0.5 * 
                ((row["HFGA_Q4"] + row["HTPA_Q4"] + (0.4 * row["HFTA_Q4"]) - (1.07 * row["HOR_Q4"]) + row["HTO_Q4"]) 
                 + (row["AFGA_Q4"] + row["ATPA_Q4"] + (0.4 * row["HFTA_Q4"]) - (1.07 * row["AOR_Q4"]) + row["ATO_Q4"]))
             )))
        dataset["DRATING"][row["AWAY"]].append(
            100 * (row["HFINAL"] / 
            (0.5 * 
                ((row["HFGA_Q4"] + row["HTPA_Q4"] + (0.4 * row["HFTA_Q4"]) - (1.07 * row["HOR_Q4"]) + row["HTO_Q4"]) 
                 + (row["AFGA_Q4"] + row["ATPA_Q4"] + (0.4 * row["HFTA_Q4"]) - (1.07 * row["AOR_Q4"]) + row["ATO_Q4"]))
             )))

        dataset["TCP"][row["HOME"]].append(
            100 * row["HFINAL"] /
            (2 * (row["HFGA_Q4"] + row["HTPA_Q4"] + (0.44 * row["HFTA_Q4"])))
        )
        dataset["TCP"][row["AWAY"]].append(
            100 * row["AFINAL"] /
            (2 * (row["AFGA_Q4"] + row["ATPA_Q4"] + (0.44 * row["AFTA_Q4"])))
        )

        dataset["APCT"][row["HOME"]].append(
            100 * (row["HAS_Q4"] /
            (row["HFGM_Q4"] + row["HTPM_Q4"])
        ))
        dataset["APCT"][row["AWAY"]].append(
            100 * (row["AAS_Q4"] /
            (row["AFGM_Q4"] + row["ATPM_Q4"])
        ))

        dataset["TOR"][row["HOME"]].append(
            100 * (row["HTO_Q4"] / (
                0.5 * ((row["HFGA_Q4"] + row["HTPA_Q4"] + (0.4 * row["HFTA_Q4"]) - (1.07 * row["HOR_Q4"])) + row["HTO_Q4"])
            ))
        )
        dataset["TOR"][row["AWAY"]].append(
            100 * (row["ATO_Q4"] / (
                0.5 * ((row["AFGA_Q4"] + row["ATPA_Q4"] + (0.4 * row["AFTA_Q4"]) - (1.07 * row["AOR_Q4"])) + row["ATO_Q4"])
            ))
        )
        
        return row
    
    print("Metrics")
    live_df = live_df.progress_apply(all_stats, axis=1)

    ratings = ["2012-01-01" for _ in range(30)]
    def add_dates(row):
        nonlocal ratings
        
        row["HREST"]            = min((date(*map(int, row["DATE"].split("-"))) - date(*map(int, ratings[row["HOME"]].split("-")))).days - 1, 14)
        row["AREST"]            = min((date(*map(int, row["DATE"].split("-"))) - date(*map(int, ratings[row["AWAY"]].split("-")))).days - 1, 14)
        ratings[row["HOME"]]    = row["DATE"]
        ratings[row["AWAY"]]    = row["DATE"]
        
        return row
    
    live_df = live_df.apply(add_dates, axis=1)
    
    live_df = live_df.copy()

    live_df['ODDS'] = 1 / (1 + 10 ** (-(live_df['HELO'] - live_df['AELO']) / 400))

    live_df['HPROJ_Q1'] = live_df['HPTS_Q1'] * 4
    live_df['APROJ_Q1'] = live_df['APTS_Q1'] * 4
    live_df['HPROJ_Q2'] = live_df['HPTS_Q2'] * 2
    live_df['APROJ_Q2'] = live_df['APTS_Q2'] * 2
    live_df['HPROJ_Q3'] = live_df['HPTS_Q3'] * 0.75
    live_df['APROJ_Q3'] = live_df['APTS_Q3'] * 0.75

    live_df["HPACE_Q1"] = (live_df["HFGA_Q1"] + live_df["HTPA_Q1"] + (0.44 * live_df["HFTA_Q1"]) - live_df["HOR_Q1"] + live_df["HTO_Q1"]) / 12
    live_df["APACE_Q1"] = (live_df["AFGA_Q1"] + live_df["ATPA_Q1"] + (0.44 * live_df["AFTA_Q1"]) - live_df["AOR_Q1"] + live_df["ATO_Q1"]) / 12
    live_df["HPACE_Q2"] = (live_df["HFGA_Q2"] + live_df["HTPA_Q2"] + (0.44 * live_df["HFTA_Q2"]) - live_df["HOR_Q2"] + live_df["HTO_Q2"]) / 24
    live_df["APACE_Q2"] = (live_df["AFGA_Q2"] + live_df["ATPA_Q2"] + (0.44 * live_df["AFTA_Q2"]) - live_df["AOR_Q2"] + live_df["ATO_Q2"]) / 24
    live_df["HPACE_Q3"] = (live_df["HFGA_Q3"] + live_df["HTPA_Q3"] + (0.44 * live_df["HFTA_Q3"]) - live_df["HOR_Q3"] + live_df["HTO_Q3"]) / 36
    live_df["APACE_Q3"] = (live_df["AFGA_Q3"] + live_df["ATPA_Q3"] + (0.44 * live_df["AFTA_Q3"]) - live_df["AOR_Q3"] + live_df["ATO_Q3"]) / 36

    with open("stds/ratings.json", "r") as file:
        all_ratings = json.loads(file.read())
    with open("stds/ratings.json", "w") as file:
        for rating in dataset:
            all_ratings[rating] = {
                team: avg_func(dataset[rating][team])
                for team in range(30)
            }
            all_ratings["LAST_GAME"] = {
                team: ratings[team]
                for team in range(30)
            }
        file.write(json.dumps(all_ratings, indent=4))

    return live_df

if __name__ == "__main__":
    live_df = pd.read_parquet("datasets/live.parquet")
    live_df = update_metrics(live_df)
    live_df.to_parquet("datasets/live.parquet")
    compose.main()
