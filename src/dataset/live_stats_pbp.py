from nba_api.live.nba.endpoints import playbyplay
import pandas as pd
import json


def get_stats(data, home_v, away_v, period, minute=0):
    minstr = ''
    if minute != 0:
        minstr = f"_M{minute}"

    stats = {
        stat: 0
        for stat in [
            f"AFGM_Q{period}{minstr}",
            f"HFGM_Q{period}{minstr}",
            f"ATPM_Q{period}{minstr}",
            f"HTPM_Q{period}{minstr}",
            f"AFTM_Q{period}{minstr}",
            f"HFTM_Q{period}{minstr}",
            f"AFGA_Q{period}{minstr}",
            f"HFGA_Q{period}{minstr}",
            f"ATPA_Q{period}{minstr}",
            f"HTPA_Q{period}{minstr}",
            f"AFTA_Q{period}{minstr}",
            f"HFTA_Q{period}{minstr}",
            f"AOR_Q{period}{minstr}",
            f"HOR_Q{period}{minstr}",
            f"ADR_Q{period}{minstr}",
            f"HDR_Q{period}{minstr}",
            f"ATR_Q{period}{minstr}",
            f"HTR_Q{period}{minstr}",
            f"AAS_Q{period}{minstr}",
            f"HAS_Q{period}{minstr}",
            f"AST_Q{period}{minstr}",
            f"HST_Q{period}{minstr}",
            f"ABLK_Q{period}{minstr}",
            f"HBLK_Q{period}{minstr}",
            f"ATO_Q{period}{minstr}",
            f"HTO_Q{period}{minstr}",
            f"APTS_Q{period}{minstr}",
            f"HPTS_Q{period}{minstr}",
            f"AFO_Q{period}{minstr}",
            f"HFO_Q{period}{minstr}",
            f"MIN_MARGIN_Q{period}{minstr}",
            f"MAX_MARGIN_Q{period}{minstr}",
        ]
    }
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
        if minute != 0 and item["period"] == period and int(item["clock"][2:4]) == minute - 1:
            break
        margin = int(item["scoreHome"]) - int(item["scoreAway"])
        min_margin = min(min_margin, margin)
        max_margin = max(max_margin, margin)
        match item["actionType"]:
            case "2pt":
                if item["shotResult"] == "Missed":
                    if item["teamId"] == home:
                        if "blockPersonId" in item:
                            stats[f"HFGA_Q{period}{minstr}"] += 1
                            stats[f"ABLK_Q{period}{minstr}"] += 1
                        else:
                            stats[f"HFGA_Q{period}{minstr}"] += 1
                    else:
                        if "blockPersonId" in item:
                            stats[f"AFGA_Q{period}{minstr}"] += 1
                            stats[f"HBLK_Q{period}{minstr}"] += 1
                        else:
                            stats[f"AFGA_Q{period}{minstr}"] += 1
                else:
                    if item["teamId"] == home:
                        if "assistTotal" in item:
                            stats[f"HAS_Q{period}{minstr}"] += 1
                            stats[f"HFGA_Q{period}{minstr}"] += 1
                            stats[f"HFGM_Q{period}{minstr}"] += 1
                            stats[f"HPTS_Q{period}{minstr}"] += 2
                        else:
                            stats[f"HFGA_Q{period}{minstr}"] += 1
                            stats[f"HFGM_Q{period}{minstr}"] += 1
                            stats[f"HPTS_Q{period}{minstr}"] += 2
                    else:
                        if "assistTotal" in item:
                            stats[f"AAS_Q{period}{minstr}"] += 1
                            stats[f"AFGA_Q{period}{minstr}"] += 1
                            stats[f"AFGM_Q{period}{minstr}"] += 1
                            stats[f"APTS_Q{period}{minstr}"] += 2
                        else:
                            stats[f"AFGA_Q{period}{minstr}"] += 1
                            stats[f"AFGM_Q{period}{minstr}"] += 1
                            stats[f"APTS_Q{period}{minstr}"] += 2
            case "3pt":
                if item["shotResult"] == "Missed":
                    if item["teamId"] == home:
                        if "blockPersonId" in item:
                            stats[f"HTPA_Q{period}{minstr}"] += 1
                            stats[f"ABLK_Q{period}{minstr}"] += 1
                        else:
                            stats[f"HTPA_Q{period}{minstr}"] += 1
                    else:
                        if "blockPersonId" in item:
                            stats[f"ATPA_Q{period}{minstr}"] += 1
                            stats[f"HBLK_Q{period}{minstr}"] += 1
                        else:
                            stats[f"ATPA_Q{period}{minstr}"] += 1
                else:
                    if item["teamId"] == home:
                        if "assistTotal" in item:
                            stats[f"HAS_Q{period}{minstr}"] += 1
                            stats[f"HTPA_Q{period}{minstr}"] += 1
                            stats[f"HTPM_Q{period}{minstr}"] += 1
                            stats[f"HPTS_Q{period}{minstr}"] += 3
                        else:
                            stats[f"HTPA_Q{period}{minstr}"] += 1
                            stats[f"HTPM_Q{period}{minstr}"] += 1
                            stats[f"HPTS_Q{period}{minstr}"] += 3
                    else:
                        if "assistTotal" in item:
                            stats[f"AAS_Q{period}{minstr}"] += 1
                            stats[f"ATPA_Q{period}{minstr}"] += 1
                            stats[f"ATPM_Q{period}{minstr}"] += 1
                            stats[f"APTS_Q{period}{minstr}"] += 3
                        else:
                            stats[f"ATPA_Q{period}{minstr}"] += 1
                            stats[f"ATPM_Q{period}{minstr}"] += 1
                            stats[f"APTS_Q{period}{minstr}"] += 3
            case "rebound":
                if len(item["qualifiers"]) == 0:
                    if item["subType"] == "offensive":
                        if item["teamId"] == home:
                            stats[f"HOR_Q{period}{minstr}"] += 1
                            stats[f"HTR_Q{period}{minstr}"] += 1
                        else:
                            stats[f"AOR_Q{period}{minstr}"] += 1
                            stats[f"ATR_Q{period}{minstr}"] += 1
                    else:
                        if item["teamId"] == home:
                            stats[f"HDR_Q{period}{minstr}"] += 1
                            stats[f"HTR_Q{period}{minstr}"] += 1
                        else:
                            stats[f"ADR_Q{period}{minstr}"] += 1
                            stats[f"ATR_Q{period}{minstr}"] += 1
            case "turnover":
                if item["teamId"] == home:
                    stats[f"HTO_Q{period}{minstr}"] += 1
                else:
                    stats[f"ATO_Q{period}{minstr}"] += 1
            case "steal":
                if item["teamId"] == home:
                    stats[f"HST_Q{period}{minstr}"] += 1
                else:
                    stats[f"AST_Q{period}{minstr}"] += 1
            case "foul":
                if item["teamId"] == home:
                    stats[f"HFO_Q{period}{minstr}"] += 1
                else:
                    stats[f"AFO_Q{period}{minstr}"] += 1
            case "freethrow":
                if item["teamId"] == home:
                    if item["shotResult"] == "Made":
                        stats[f"HFTA_Q{period}{minstr}"] += 1
                        stats[f"HFTM_Q{period}{minstr}"] += 1
                        stats[f"HPTS_Q{period}{minstr}"] += 1
                    else:
                        stats[f"HFTA_Q{period}{minstr}"] += 1
                else:
                    if item["shotResult"] == "Made":
                        stats[f"AFTA_Q{period}{minstr}"] += 1
                        stats[f"AFTM_Q{period}{minstr}"] += 1
                        stats[f"APTS_Q{period}{minstr}"] += 1
                    else:
                        stats[f"AFTA_Q{period}{minstr}"] += 1
            case "period":
                if item["subType"] == "end" and item["period"] == period:
                    break
    stats[f"MIN_MARGIN_Q{period}{minstr}"] = min_margin
    stats[f"MAX_MARGIN_Q{period}{minstr}"] = max_margin
    factor = [0, 4, 2, 0.75]
    if minute == 0:
        stats[f"HPROJ_Q{period}"] = stats[f"HPTS_Q{period}"] * factor[period]
        stats[f"APROJ_Q{period}"] = stats[f"APTS_Q{period}"] * factor[period]
    return stats
