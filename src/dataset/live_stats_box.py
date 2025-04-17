from nba_api.live.nba.endpoints import boxscore
import pandas as pd


def live_stats_box(idx):

    pbp = boxscore.BoxScore(idx)
    df = pbp.get_dict()["game"]

    return {
        "AFGM": df["awayTeam"]["statistics"]["twoPointersMade"],
        "HFGM": df["homeTeam"]["statistics"]["twoPointersMade"],
        "ATPM": df["awayTeam"]["statistics"]["threePointersMade"],
        "HTPM": df["homeTeam"]["statistics"]["threePointersMade"],
        "AFTM": df["awayTeam"]["statistics"]["freeThrowsMade"],
        "HFTM": df["homeTeam"]["statistics"]["freeThrowsMade"],
        "AFGA": df["awayTeam"]["statistics"]["twoPointersAttempted"],
        "HFGA": df["homeTeam"]["statistics"]["twoPointersAttempted"],
        "ATPA": df["awayTeam"]["statistics"]["threePointersAttempted"],
        "HTPA": df["homeTeam"]["statistics"]["threePointersAttempted"],
        "AFTA": df["awayTeam"]["statistics"]["freeThrowsAttempted"],
        "HFTA": df["homeTeam"]["statistics"]["freeThrowsAttempted"],
        "AOR": df["awayTeam"]["statistics"]["reboundsOffensive"],
        "HOR": df["homeTeam"]["statistics"]["reboundsOffensive"],
        "ADR": df["awayTeam"]["statistics"]["reboundsDefensive"],
        "HDR": df["homeTeam"]["statistics"]["reboundsDefensive"],
        "ATR": df["awayTeam"]["statistics"]["reboundsTotal"]
        - df["awayTeam"]["statistics"]["reboundsTeam"],
        "HTR": df["homeTeam"]["statistics"]["reboundsTotal"]
        - df["homeTeam"]["statistics"]["reboundsTeam"],
        "AAS": df["awayTeam"]["statistics"]["assists"],
        "HAS": df["homeTeam"]["statistics"]["assists"],
        "AST": df["awayTeam"]["statistics"]["steals"],
        "HST": df["homeTeam"]["statistics"]["steals"],
        "ABLK": df["awayTeam"]["statistics"]["blocks"],
        "HBLK": df["homeTeam"]["statistics"]["blocks"],
        "ATO": df["awayTeam"]["statistics"]["turnoversTotal"],
        "HTO": df["homeTeam"]["statistics"]["turnoversTotal"],
        "APTS": df["awayTeam"]["statistics"]["points"],
        "HPTS": df["homeTeam"]["statistics"]["points"],
        "AFO": df["awayTeam"]["statistics"]["foulsPersonal"],
        "HFO": df["homeTeam"]["statistics"]["foulsPersonal"],
    }
