import pandas as pd
import json
from collections import defaultdict


def live_player_pbp(data, home_v, away_v,):
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

    
    home = 5
    away = 5

    poss = True

    hf, af = False, False
    for item in data["game"]["actions"]:
        if "teamTricode" in item:
            if item["teamTricode"] == home_v:
                home = item["teamId"]
                hf = True
            elif item["teamTricode"] == away_v:
                away = item["teamId"]
                af = True
        if hf and af:
            break

    min_margin = 0
    max_margin = 0
    for item in data["game"]["actions"]:
        if item["actionType"] not in { "2pt", "3pt", "rebound", "turnover", "steal", "foul", "freethrow" } and stats[item["personId"]]["out"]:
            stats[item["personId"]]["out"] = False
            stats[item["personId"]]["inTime"] = 12 * (5 - item["period"])
        if "assistPersonId" in item and stats[item["assistPersonId"]]["out"] == True:
            stats[item["assistPersonId"]]["out"] = False
            stats[item["assistPersonId"]]["inTime"] = 12 * (5 - item["period"])
        if "blockPersonId" in item and stats[item["blockPersonId"]]["out"] == True:
            stats[item["blockPersonId"]]["out"] = False
            stats[item["blockPersonId"]]["inTime"] = 12 * (5 - item["period"])
        
        match item["actionType"]:
            case "2pt":
                if item["shotResult"] == "Missed":
                    if item["teamId"] == home:
                        if "blockPersonId" in item:
                            for q in range(item['period'], 5):
                                stats[item["personId"]][f"FGA_Q{q}"] += 1
                                stats[item["personId"]]["team"] = True
                                stats[item["blockPersonId"]][f"BLK_Q{q}"] += 1
                                stats[item["blockPersonId"]]["team"] = False
                        else:
                            for q in range(item['period'], 5):
                                stats[item["personId"]][f"FGA_Q{q}"] += 1
                                stats[item["personId"]]["team"] = True
                    else:
                        if "blockPersonId" in item:
                            for q in range(item['period'], 5):
                                stats[item["personId"]][f"FGA_Q{q}"] += 1
                                stats[item["personId"]]["team"] = False
                                stats[item["blockPersonId"]][f"BLK_Q{q}"] += 1
                                stats[item["blockPersonId"]]["team"] = True
                        else:
                            for q in range(item['period'], 5):
                                stats[item["personId"]][f"FGA_Q{q}"] += 1
                                stats[item["personId"]]["team"] = False
                else:
                    if item["teamId"] == home:
                        if "assistTotal" in item:
                            for q in range(item['period'], 5):
                                stats[item["personId"]][f"FGA_Q{q}"] += 1
                                stats[item["personId"]][f"FGM_Q{q}"] += 1
                                stats[item["personId"]][f"PTS_Q{q}"] += 2
                                stats[item["personId"]]["team"] = True
                                stats[item["assistPersonId"]][f"AS_Q{q}"] += 1
                                stats[item["assistPersonId"]]["team"] = True
                        else:
                            for q in range(item['period'], 5):
                                stats[item["personId"]][f"FGA_Q{q}"] += 1
                                stats[item["personId"]][f"FGM_Q{q}"] += 1
                                stats[item["personId"]][f"PTS_Q{q}"] += 2
                                stats[item["personId"]]["team"] = True
                    else:
                        if "assistTotal" in item:
                            for q in range(item['period'], 5):
                                stats[item["personId"]][f"FGA_Q{q}"] += 1
                                stats[item["personId"]][f"FGM_Q{q}"] += 1
                                stats[item["personId"]][f"PTS_Q{q}"] += 2
                                stats[item["personId"]]["team"] = False
                                stats[item["assistPersonId"]][f"AS_Q{q}"] += 1
                                stats[item["assistPersonId"]]["team"] = False
                        else:
                            for q in range(item['period'], 5):
                                stats[item["personId"]][f"FGA_Q{q}"] += 1
                                stats[item["personId"]][f"FGM_Q{q}"] += 1
                                stats[item["personId"]][f"PTS_Q{q}"] += 2
                                stats[item["personId"]]["team"] = False
            case "3pt":
                if item["shotResult"] == "Missed":
                    if item["teamId"] == home:
                        if "blockPersonId" in item:
                            for q in range(item['period'], 5):
                                stats[item["personId"]][f"TPA_Q{q}"] += 1
                                stats[item["personId"]]["team"] = True
                                stats[item["blockPersonId"]][f"BLK_Q{q}"] += 1
                                stats[item["blockPersonId"]]["team"] = False
                        else:
                            for q in range(item['period'], 5):
                                stats[item["personId"]][f"TPA_Q{q}"] += 1
                                stats[item["personId"]]["team"] = True
                    else:
                        if "blockPersonId" in item:
                            for q in range(item['period'], 5):
                                stats[item["personId"]][f"TPA_Q{q}"] += 1
                                stats[item["personId"]]["team"] = False
                                stats[item["blockPersonId"]][f"BLK_Q{q}"] += 1
                                stats[item["blockPersonId"]]["team"] = True
                        else:
                            for q in range(item['period'], 5):
                                stats[item["personId"]][f"FGA_Q{q}"] += 1
                                stats[item["personId"]]["team"] = False
                else:
                    if item["teamId"] == home:
                        if "assistTotal" in item:
                            for q in range(item['period'], 5):
                                stats[item["personId"]][f"TPA_Q{q}"] += 1
                                stats[item["personId"]][f"TPM_Q{q}"] += 1
                                stats[item["personId"]][f"PTS_Q{q}"] += 3
                                stats[item["personId"]]["team"] = True
                                stats[item["assistPersonId"]][f"AS_Q{q}"] += 1
                                stats[item["assistPersonId"]]["team"] = True
                        else:
                            for q in range(item['period'], 5):
                                stats[item["personId"]][f"TPA_Q{q}"] += 1
                                stats[item["personId"]][f"TPM_Q{q}"] += 1
                                stats[item["personId"]][f"PTS_Q{q}"] += 3
                                stats[item["personId"]]["team"] += True
                    else:
                        if "assistTotal" in item:
                            for q in range(item['period'], 5):
                                stats[item["personId"]][f"TPA_Q{q}"] += 1
                                stats[item["personId"]][f"TPM_Q{q}"] += 1
                                stats[item["personId"]][f"PTS_Q{q}"] += 3
                                stats[item["personId"]]["team"] = False
                                stats[item["assistPersonId"]][f"AS_Q{q}"] += 1
                                stats[item["assistPersonId"]]["team"] = False
                        else:
                            for q in range(item['period'], 5):
                                stats[item["personId"]][f"TPA_Q{q}"] += 1
                                stats[item["personId"]][f"TPM_Q{q}"] += 1
                                stats[item["personId"]][f"PTS_Q{q}"] += 3
                                stats[item["personId"]]["team"] = False
            case "rebound":
                if len(item["qualifiers"]) == 0:
                    if item["subType"] == "offensive":
                        if item["teamId"] == home:
                            for q in range(item['period'], 5):
                                stats[item["personId"]][f"OR_Q{q}"] += 1
                                stats[item["personId"]][f"TR_Q{q}"] += 1
                                stats[item["personId"]]["team"] = True
                        else:
                            for q in range(item['period'], 5):
                                stats[item["personId"]][f"OR_Q{q}"] += 1
                                stats[item["personId"]][f"TR_Q{q}"] += 1
                                stats[item["personId"]]["team"] = False
                    else:
                        if item["teamId"] == home:
                            for q in range(item['period'], 5):
                                stats[item["personId"]][f"DR_Q{q}"] += 1
                                stats[item["personId"]][f"TR_Q{q}"] += 1
                                stats[item["personId"]]["team"] = True
                        else:
                            for q in range(item['period'], 5):
                                stats[item["personId"]][f"DR_Q{q}"] += 1
                                stats[item["personId"]][f"TR_Q{q}"] += 1
                                stats[item["personId"]]["team"] = False
            case "turnover":
                if item["teamId"] == home:
                    for q in range(item['period'], 5):
                        stats[item["personId"]][f"TO_Q{q}"] += 1
                        stats[item["personId"]]["team"] = True
                else:
                    for q in range(item['period'], 5):
                        stats[item["personId"]][f"TO_Q{q}"] += 1
                        stats[item["personId"]]["team"] = False
            case "steal":
                if item["teamId"] == home:
                    for q in range(item['period'], 5):
                        stats[item["personId"]][f"ST_Q{q}"] += 1
                        stats[item["personId"]]["team"] = True
                else:
                    for q in range(item['period'], 5):
                        stats[item["personId"]][f"ST_Q{q}"] += 1
                        stats[item["personId"]]["team"] = False
            case "foul":
                if item["teamId"] == home:
                    for q in range(item['period'], 5):
                        stats[item["personId"]][f"FO_Q{q}"] += 1
                        stats[item["personId"]]["team"] = True
                else:
                    for q in range(item['period'], 5):
                        stats[item["personId"]][f"FO_Q{q}"] += 1
                        stats[item["personId"]]["team"] = False
            case "freethrow":
                if item["teamId"] == home:
                    if item["shotResult"] == "Made":
                        for q in range(item['period'], 5):
                            stats[item["personId"]][f"FTA_Q{q}"] += 1
                            stats[item["personId"]][f"FTM_Q{q}"] += 1
                            stats[item["personId"]][f"PTS_Q{q}"] += 1
                            stats[item["personId"]]["team"] = True
                    else:
                        for q in range(item['period'], 5):
                            stats[item["personId"]][f"FTA_Q{q}"] += 1
                            stats[item["personId"]]["team"] = True
                else:
                    if item["shotResult"] == "Made":
                        for q in range(item['period'], 5):
                            stats[item["personId"]][f"FTA_Q{q}"] += 1
                            stats[item["personId"]][f"FTM_Q{q}"] += 1
                            stats[item["personId"]][f"PTS_Q{q}"] += 1
                            stats[item["personId"]]["team"] = False
                    else:
                        for q in range(item['period'], 5):
                            stats[item["personId"]][f"FTA_Q{q}"] += 1
                            stats[item["personId"]]["team"] = False
            case "substitution":
                minutes = item["clock"][2:-1]
                minutes = (12 * (4 - item["period"])) + int(minutes.split("M")[0]) + (float(minutes.split("M")[1]) / 60)
                if item["subType"] == "in":
                    stats[item["personId"]]["out"] = False
                    stats[item["personId"]]["inTime"] = minutes
                else:
                    stats[item["personId"]]["out"] = True
                    stats[item["personId"]]["MINUTES"] += stats[item["personId"]]["inTime"] - minutes
            case "period":
                if item["subType"] == "end":
                    for player in stats:
                        if not stats[player]["out"]:
                            minutes = 12 * (4 - item["period"])
                            stats[player]["out"] = True
                            stats[player]["MINUTES"] += stats[player]["inTime"] - minutes
        
    for player in stats:
        if not stats[player]["out"]:
            stats[player]["MINUTES"] += stats[player]["inTime"]
        
        stats[player].pop("out")
        stats[player].pop("inTime")

    return dict(stats)

if __name__ == "__main__":
    from nba_api.live.nba.endpoints import playbyplay
    import json
    data = playbyplay.PlayByPlay("0022400585").get_dict()
    stats = live_player_pbp(data, "LAL", "BKN")
    print(json.dumps(stats, indent=4))