import pandas as pd
from collections import defaultdict


def player_pbp_data(df):
    stats = defaultdict(lambda : {
    "out": False,
    "inTime": 48.,
    "MINUTES": 0,
    "team": None,
    **{
        f"{stat}_Q{q}": 0
        for q in range(1,5)
        for stat in [
            "TPA",
            "TPM",
            "FGA",
            "FGM",
            "FTA",
            "FTM",
            "FO",
            "OR",
            "DR",
            "TR",
            "AS",
            "ST",
            "BLK",
            "TO",
            "PTS",
        ]
    }})

    poss = True
    i = 0


    for idx, row in df.iterrows():
        stats[row["PLAYER1_ID"]]["name"] = row["PLAYER1_NAME"]
        if row["EVENTMSGTYPE"] in range(1, 7):
            if stats[row["PLAYER1_ID"]]["out"]:
                stats[row["PLAYER1_ID"]]["out"] = False
                stats[row["PLAYER1_ID"]]["inTime"] = 12 * (5 - row["PERIOD"])

            if row["PLAYER2_ID"] and stats[row["PLAYER2_ID"]]["out"]:
                stats[row["PLAYER2_ID"]]["out"] = False
                stats[row["PLAYER2_ID"]]["inTime"] = 12 * (5 - row["PERIOD"])
            
        match row["EVENTMSGTYPE"]:
            case 1:  # FG made
                if row["HOMEDESCRIPTION"]:
                    stats[row["PLAYER1_ID"]]["team"] = True
                    poss = True
                    if "AST" in row["HOMEDESCRIPTION"]:
                        for q in range(row["PERIOD"], 5):
                            stats[row["PLAYER2_ID"]][f"AS_Q{q}"] += 1
                            stats[row["PLAYER2_ID"]]["team"] = True
                    if "3PT" in row["HOMEDESCRIPTION"]:
                        for q in range(row["PERIOD"], 5):
                            stats[row["PLAYER1_ID"]][f"TPA_Q{q}"] += 1
                            stats[row["PLAYER1_ID"]][f"TPM_Q{q}"] += 1
                            stats[row["PLAYER1_ID"]][f"PTS_Q{q}"] += 3
                    else:
                        for q in range(row["PERIOD"], 5):
                            stats[row["PLAYER1_ID"]][f"FGA_Q{q}"] += 1
                            stats[row["PLAYER1_ID"]][f"FGM_Q{q}"] += 1
                            stats[row["PLAYER1_ID"]][f"PTS_Q{q}"] += 2
                else:
                    stats[row["PLAYER1_ID"]]["team"] = False
                    poss = False
                    if "AST" in row["VISITORDESCRIPTION"]:
                        for q in range(row["PERIOD"], 5):
                            stats[row["PLAYER2_ID"]][f"AS_Q{q}"] += 1
                            stats[row["PLAYER2_ID"]]["team"] = False
                    if "3PT" in row["VISITORDESCRIPTION"]:
                        for q in range(row["PERIOD"], 5):
                            stats[row["PLAYER1_ID"]][f"TPA_Q{q}"] += 1
                            stats[row["PLAYER1_ID"]][f"TPM_Q{q}"] += 1
                            stats[row["PLAYER1_ID"]][f"PTS_Q{q}"] += 3
                    else:
                        for q in range(row["PERIOD"], 5):
                            stats[row["PLAYER1_ID"]][f"FGA_Q{q}"] += 1
                            stats[row["PLAYER1_ID"]][f"FGM_Q{q}"] += 1
                            stats[row["PLAYER1_ID"]][f"PTS_Q{q}"] += 2
            case 2:  # FG missed
                if row["HOMEDESCRIPTION"] and "BLOCK" in row["HOMEDESCRIPTION"]:
                    poss = False
                    if row["VISITORDESCRIPTION"] and "3PT" in row["VISITORDESCRIPTION"]:
                        for q in range(row["PERIOD"], 5):
                            stats[row["PLAYER3_ID"]]["team"] = True
                            stats[row["PLAYER1_ID"]]["team"] = False
                            stats[row["PLAYER1_ID"]][f"TPA_Q{q}"] += 1
                            stats[row["PLAYER3_ID"]][f"BLK_Q{q}"] += 1
                    else:
                        for q in range(row["PERIOD"], 5):
                            stats[row["PLAYER3_ID"]]["team"] = True
                            stats[row["PLAYER1_ID"]]["team"] = False
                            stats[row["PLAYER1_ID"]][f"FGA_Q{q}"] += 1
                            stats[row["PLAYER3_ID"]][f"BLK_Q{q}"] += 1
                elif row["VISITORDESCRIPTION"] and "BLOCK" in row["VISITORDESCRIPTION"]:
                    poss = True
                    if row["HOMEDESCRIPTION"] and "3PT" in row["HOMEDESCRIPTION"]:
                        for q in range(row["PERIOD"], 5):
                            stats[row["PLAYER3_ID"]]["team"] = False
                            stats[row["PLAYER1_ID"]]["team"] = True
                            stats[row["PLAYER1_ID"]][f"TPA_Q{q}"] += 1
                            stats[row["PLAYER3_ID"]][f"BLK_Q{q}"] += 1
                    else:
                        for q in range(row["PERIOD"], 5):
                            stats[row["PLAYER3_ID"]]["team"] = False
                            stats[row["PLAYER1_ID"]]["team"] = True
                            stats[row["PLAYER1_ID"]][f"FGA_Q{q}"] += 1
                            stats[row["PLAYER3_ID"]][f"BLK_Q{q}"] += 1
                elif row["HOMEDESCRIPTION"]:
                    poss = True
                    if "3PT" in row["HOMEDESCRIPTION"]:
                        for q in range(row["PERIOD"], 5):
                            stats[row["PLAYER1_ID"]]["team"] = True
                            stats[row["PLAYER1_ID"]][f"TPA_Q{q}"] += 1
                    else:
                        for q in range(row["PERIOD"], 5):
                            stats[row["PLAYER1_ID"]]["team"] = True
                            stats[row["PLAYER1_ID"]][f"FGA_Q{q}"] += 1
                else:
                    poss = False
                    if "3PT" in row["VISITORDESCRIPTION"]:
                        for q in range(row["PERIOD"], 5):
                            stats[row["PLAYER1_ID"]]["team"] = False
                            stats[row["PLAYER1_ID"]][f"TPA_Q{q}"] += 1
                    else:
                        for q in range(row["PERIOD"], 5):
                            stats[row["PLAYER1_ID"]]["team"] = False
                            stats[row["PLAYER1_ID"]][f"FGA_Q{q}"] += 1
            case 3:  # FT
                if row["HOMEDESCRIPTION"]:
                    stats[row["PLAYER1_ID"]]["team"] = True
                    poss = True
                    if "MISS" in row["HOMEDESCRIPTION"]:
                        for q in range(row["PERIOD"], 5):
                            stats[row["PLAYER1_ID"]][f"FTA_Q{q}"] += 1
                    else:
                        for q in range(row["PERIOD"], 5):
                            stats[row["PLAYER1_ID"]][f"FTA_Q{q}"] += 1
                            stats[row["PLAYER1_ID"]][f"FTM_Q{q}"] += 1
                            stats[row["PLAYER1_ID"]][f"PTS_Q{q}"] += 1
                else:
                    stats[row["PLAYER1_ID"]]["team"] = False
                    poss = False
                    if "MISS" in row["VISITORDESCRIPTION"]:
                        for q in range(row["PERIOD"], 5):
                            stats[row["PLAYER1_ID"]][f"FTA_Q{q}"] += 1
                    else:
                        for q in range(row["PERIOD"], 5):
                            stats[row["PLAYER1_ID"]][f"FTA_Q{q}"] += 1
                            stats[row["PLAYER1_ID"]][f"FTM_Q{q}"] += 1
                            stats[row["PLAYER1_ID"]][f"PTS_Q{q}"] += 1
            case 4:
                if row["HOMEDESCRIPTION"] and "REBOUND" in row["HOMEDESCRIPTION"]:
                    stats[row["PLAYER1_ID"]]["team"] = True
                    if poss:
                        for q in range(row["PERIOD"], 5):
                            stats[row["PLAYER1_ID"]][f"OR_Q{q}"] += 1
                            stats[row["PLAYER1_ID"]][f"TR_Q{q}"] += 1
                    else:
                        for q in range(row["PERIOD"], 5):
                            stats[row["PLAYER1_ID"]][f"DR_Q{q}"] += 1
                            stats[row["PLAYER1_ID"]][f"TR_Q{q}"] += 1
                elif (
                    row["VISITORDESCRIPTION"] and "REBOUND" in row["VISITORDESCRIPTION"]
                ):
                    stats[row["PLAYER1_ID"]]["team"] = False
                    if poss:
                        for q in range(row["PERIOD"], 5):
                            stats[row["PLAYER1_ID"]][f"DR_Q{q}"] += 1
                            stats[row["PLAYER1_ID"]][f"TR_Q{q}"] += 1
                    else:
                        for q in range(row["PERIOD"], 5):
                            stats[row["PLAYER1_ID"]][f"OR_Q{q}"] += 1
                            stats[row["PLAYER1_ID"]][f"TR_Q{q}"] += 1
            case 5:
                if row["HOMEDESCRIPTION"] and "STEAL" in row["HOMEDESCRIPTION"]:
                    for q in range(row["PERIOD"], 5):
                        stats[row["PLAYER1_ID"]][f"ST_Q{q}"] += 1
                        stats[row["PLAYER2_ID"]][f"TO_Q{q}"] += 1
                elif row["VISITORDESCRIPTION"] and "STEAL" in row["VISITORDESCRIPTION"]:
                    for q in range(row["PERIOD"], 5):
                        stats[row["PLAYER2_ID"]][f"ST_Q{q}"] += 1
                        stats[row["PLAYER1_ID"]][f"TO_Q{q}"] += 1
                elif row["HOMEDESCRIPTION"]:
                    for q in range(row["PERIOD"], 5):
                        stats[row["PLAYER1_ID"]][f"TO_Q{q}"] += 1
                elif row["VISITORDESCRIPTION"]:
                    for q in range(row["PERIOD"], 5):
                        stats[row["PLAYER1_ID"]][f"TO_Q{q}"] += 1
            case 6:
                if row["HOMEDESCRIPTION"]:
                    for q in range(row["PERIOD"], 5):
                        stats[row["PLAYER1_ID"]][f"FO_Q{q}"] += 1
                elif row["VISITORDESCRIPTION"]:
                    for q in range(row["PERIOD"], 5):
                        stats[row["PLAYER1_ID"]][f"FO_Q{q}"] += 1
            case 8:
                minutes = (12 * (4 - row["PERIOD"])) + int(row["PCTIMESTRING"].split(":")[0]) + (int(row["PCTIMESTRING"].split(":")[1]) / 60)
                stats[row["PLAYER2_ID"]]["out"] = False
                stats[row["PLAYER2_ID"]]["inTime"] = minutes

                stats[row["PLAYER1_ID"]]["out"] = True
                stats[row["PLAYER1_ID"]]["MINUTES"] += stats[row["PLAYER1_ID"]]["inTime"] - minutes
            case 13:
                for player in stats:
                    if not stats[player]["out"]:
                        minutes = 12 * (4 - row["PERIOD"])

                        stats[player]["out"] = True
                        stats[player]["MINUTES"] += stats[player]["inTime"] - minutes

    for player in stats:
        if not stats[player]["out"]:
            stats[player]["MINUTES"] += stats[player]["inTime"]
        
        stats[player].pop("out")
        stats[player].pop("inTime")

    return dict(stats)

if __name__ == "__main__":
    import json
    from nba_api.stats.endpoints import playbyplayv2
    data = playbyplayv2.PlayByPlayV2("0022400585").get_data_frames()[0]
    stats = player_pbp_data(data)
    print(json.dumps(stats,indent=4))