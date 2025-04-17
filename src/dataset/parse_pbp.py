import pandas as pd


def get_pbp_data(df):
    timestrs = ["Q1_M8", "Q1_M6", "Q1_M4", "Q1", "Q2_M8", "Q2_M6", "Q2_M4", "Q2", "Q3_M8", "Q3_M6", "Q3_M4", "Q3", "Q4_M8", "Q4_M6", "Q4_M4", "Q4"]

    stats = {
        stat: 0
        for timestr in timestrs
        for stat in [
            f"AFGM_{timestr}",
            f"HFGM_{timestr}",
            f"ATPM_{timestr}",
            f"HTPM_{timestr}",
            f"AFTM_{timestr}",
            f"HFTM_{timestr}",
            f"AFGA_{timestr}",
            f"HFGA_{timestr}",
            f"ATPA_{timestr}",
            f"HTPA_{timestr}",
            f"AFTA_{timestr}",
            f"HFTA_{timestr}",
            f"AOR_{timestr}",
            f"HOR_{timestr}",
            f"ADR_{timestr}",
            f"HDR_{timestr}",
            f"ATR_{timestr}",
            f"HTR_{timestr}",
            f"AAS_{timestr}",
            f"HAS_{timestr}",
            f"AST_{timestr}",
            f"HST_{timestr}",
            f"ABLK_{timestr}",
            f"HBLK_{timestr}",
            f"ATO_{timestr}",
            f"HTO_{timestr}",
            f"APTS_{timestr}",
            f"HPTS_{timestr}",
            f"AFO_{timestr}",
            f"HFO_{timestr}",
            f"MIN_MARGIN_{timestr}",
            f"MAX_MARGIN_{timestr}",
        ]
    }
    stats["MIN"] = 48

    poss = True
    i = 0

    pmin = 12

    pp = 1

    for idx, row in df.iterrows():
        if row["PERIOD"] != pp:
            if pp >= 4:
                stats["MIN"] += 5 # busted, just adds min for overtime
            pp = row["PERIOD"]
        curmin = int(row["PCTIMESTRING"].split(":")[0])
        if pmin != curmin and curmin + 1 in {4, 6, 8}:
            i += 1
        pmin = curmin
        match row["EVENTMSGTYPE"]:
            case 1:  # FG made
                try:
                    cur_margin = int(row["SCOREMARGIN"])
                except:
                    pass
                else:
                    for timestr in timestrs[i:]:
                        stats[f"MIN_MARGIN_{timestr}"] = min(
                            stats[f"MIN_MARGIN_{timestr}"], int(row["SCOREMARGIN"])
                        )
                        stats[f"MAX_MARGIN_{timestr}"] = max(
                            stats[f"MAX_MARGIN_{timestr}"], int(row["SCOREMARGIN"])
                        )
                if row["HOMEDESCRIPTION"]:
                    poss = True
                    if "AST" in row["HOMEDESCRIPTION"]:
                        for timestr in timestrs[i:]:
                            stats[f"HAS_{timestr}"] += 1
                    if "3PT" in row["HOMEDESCRIPTION"]:
                        for timestr in timestrs[i:]:
                            stats[f"HTPA_{timestr}"] += 1
                            stats[f"HTPM_{timestr}"] += 1
                            stats[f"HPTS_{timestr}"] += 3
                    else:
                        for timestr in timestrs[i:]:
                            stats[f"HFGA_{timestr}"] += 1
                            stats[f"HFGM_{timestr}"] += 1
                            stats[f"HPTS_{timestr}"] += 2
                else:
                    poss = False
                    if "AST" in row["VISITORDESCRIPTION"]:
                        for timestr in timestrs[i:]:
                            stats[f"AAS_{timestr}"] += 1
                    if "3PT" in row["VISITORDESCRIPTION"]:
                        for timestr in timestrs[i:]:
                            stats[f"ATPA_{timestr}"] += 1
                            stats[f"ATPM_{timestr}"] += 1
                            stats[f"APTS_{timestr}"] += 3
                    else:
                        for timestr in timestrs[i:]:
                            stats[f"AFGA_{timestr}"] += 1
                            stats[f"AFGM_{timestr}"] += 1
                            stats[f"APTS_{timestr}"] += 2
            case 2:  # FG missed
                if row["HOMEDESCRIPTION"] and "BLOCK" in row["HOMEDESCRIPTION"]:
                    poss = False
                    if row["VISITORDESCRIPTION"] and "3PT" in row["VISITORDESCRIPTION"]:
                        for timestr in timestrs[i:]:
                            stats[f"ATPA_{timestr}"] += 1
                            stats[f"HBLK_{timestr}"] += 1
                    else:
                        for timestr in timestrs[i:]:
                            stats[f"AFGA_{timestr}"] += 1
                            stats[f"HBLK_{timestr}"] += 1
                elif row["VISITORDESCRIPTION"] and "BLOCK" in row["VISITORDESCRIPTION"]:
                    poss = True
                    if row["HOMEDESCRIPTION"] and "3PT" in row["HOMEDESCRIPTION"]:
                        for timestr in timestrs[i:]:
                            stats[f"HTPA_{timestr}"] += 1
                            stats[f"ABLK_{timestr}"] += 1
                    else:
                        for timestr in timestrs[i:]:
                            stats[f"HFGA_{timestr}"] += 1
                            stats[f"ABLK_{timestr}"] += 1
                elif row["HOMEDESCRIPTION"]:
                    poss = True
                    if "3PT" in row["HOMEDESCRIPTION"]:
                        for timestr in timestrs[i:]:
                            stats[f"HTPA_{timestr}"] += 1
                    else:
                        for timestr in timestrs[i:]:
                            stats[f"HFGA_{timestr}"] += 1
                else:
                    poss = False
                    if "3PT" in row["VISITORDESCRIPTION"]:
                        for timestr in timestrs[i:]:
                            stats[f"ATPA_{timestr}"] += 1
                    else:
                        for timestr in timestrs[i:]:
                            stats[f"AFGA_{timestr}"] += 1
            case 3:  # FT
                if row["HOMEDESCRIPTION"]:
                    poss = True
                    if "MISS" in row["HOMEDESCRIPTION"]:
                        for timestr in timestrs[i:]:
                            stats[f"HFTA_{timestr}"] += 1
                    else:
                        for timestr in timestrs[i:]:
                            stats[f"HFTA_{timestr}"] += 1
                            stats[f"HFTM_{timestr}"] += 1
                            stats[f"HPTS_{timestr}"] += 1
                else:
                    poss = False
                    if "MISS" in row["VISITORDESCRIPTION"]:
                        for timestr in timestrs[i:]:
                            stats[f"AFTA_{timestr}"] += 1
                    else:
                        for timestr in timestrs[i:]:
                            stats[f"AFTA_{timestr}"] += 1
                            stats[f"AFTM_{timestr}"] += 1
                            stats[f"APTS_{timestr}"] += 1
            case 4:
                if row["HOMEDESCRIPTION"] and "REBOUND" in row["HOMEDESCRIPTION"]:
                    if poss:
                        for timestr in timestrs[i:]:
                            stats[f"HOR_{timestr}"] += 1
                    else:
                        for timestr in timestrs[i:]:
                            stats[f"HDR_{timestr}"] += 1
                    for timestr in timestrs[i:]:
                        stats[f"HTR_{timestr}"] += 1
                elif (
                    row["VISITORDESCRIPTION"] and "REBOUND" in row["VISITORDESCRIPTION"]
                ):
                    if poss:
                        for timestr in timestrs[i:]:
                            stats[f"ADR_{timestr}"] += 1
                    else:
                        for timestr in timestrs[i:]:
                            stats[f"AOR_{timestr}"] += 1
                    for timestr in timestrs[i:]:
                        stats[f"ATR_{timestr}"] += 1
            case 5:
                if row["HOMEDESCRIPTION"] and "STEAL" in row["HOMEDESCRIPTION"]:
                    for timestr in timestrs[i:]:
                        stats[f"HST_{timestr}"] += 1
                        stats[f"ATO_{timestr}"] += 1
                elif row["VISITORDESCRIPTION"] and "STEAL" in row["VISITORDESCRIPTION"]:
                    for timestr in timestrs[i:]:
                        stats[f"AST_{timestr}"] += 1
                        stats[f"HTO_{timestr}"] += 1
                elif row["HOMEDESCRIPTION"]:
                    for timestr in timestrs[i:]:
                        stats[f"HTO_{timestr}"] += 1
                elif row["VISITORDESCRIPTION"]:
                    for timestr in timestrs[i:]:
                        stats[f"ATO_{timestr}"] += 1
            case 6:
                if row["HOMEDESCRIPTION"]:
                    for timestr in timestrs[i:]:
                        stats[f"HFO_{timestr}"] += 1
                elif row["VISITORDESCRIPTION"]:
                    for timestr in timestrs[i:]:
                        stats[f"AFO_{timestr}"] += 1
            case 13:
                i += 1
    return stats

if __name__ == "__main__":
    df = pd.read_parquet("pbps/0022401221.parquet")
    print(get_pbp_data(df))