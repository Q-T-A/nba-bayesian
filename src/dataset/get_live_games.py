from nba_api.stats.endpoints import leaguegamelog
from nba_api.live.nba.endpoints import playbyplay
from datetime import date, timedelta
import pandas as pd
from live_stats_pbp import get_stats
import time


def get_live_games(prev, minutes):
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
    try:
        yesterday = date.today() - timedelta(days=1)
        tomorrow =  date.today() + timedelta(days=1)
        gamelog = leaguegamelog.LeagueGameLog(date_from_nullable=str(yesterday))
        data_frames = gamelog.get_data_frames()
    except Exception as e:
        print("Error fetching game log:", e)
        gamelog = leaguegamelog.LeagueGameLog(
        date_from_nullable=str(date.today())
        )
    gamelog = gamelog.get_data_frames()[0]
    gamelog = gamelog[gamelog["WL"] != "W"]
    gamelog = gamelog[gamelog["WL"] != "L"]
    games = []
    for idx, row in gamelog.iterrows():
        if "@" in row["MATCHUP"]:
            data = playbyplay.PlayByPlay(row["GAME_ID"]).get_dict()["game"]["actions"][-1]
            time.sleep(0.6)
            if int(data["period"]) < 4:# and data["description"] == "Period End":
                games.append(
                    {
                        "GAME_ID": row["GAME_ID"],
                        "HOME": abbrev_map[row["MATCHUP"].split()[2]],
                        "HOME_V": row["MATCHUP"].split()[2],
                        "AWAY": abbrev_map[row["MATCHUP"].split()[0]],
                        "AWAY_V": row["MATCHUP"].split()[0],
                    }
                )
    cur_df = pd.DataFrame(games)
    print(cur_df)
    def fetch_game_stats(row):       
        stats = {}
        nonlocal minutes
        nonlocal prev

        pbp = playbyplay.PlayByPlay(row["GAME_ID"])
        data = pbp.get_dict()
        period = data["game"]["actions"][-1]["period"]

        stats["period"] = period

        if period == 1:
            if minutes:
                stats = {**stats, **get_stats(data, row["HOME_V"], row["AWAY_V"], 1, 8)}
                stats = {**stats, **get_stats(data, row["HOME_V"], row["AWAY_V"], 1, 4)}
    
            stats = {**stats, **get_stats(data, row["HOME_V"], row["AWAY_V"], 1)}

        if period == 2:
            if prev:
                stats = {**stats, **get_stats(data, row["HOME_V"], row["AWAY_V"], 1)}

                if minutes:
                    stats = {**stats, **get_stats(data, row["HOME_V"], row["AWAY_V"], 1, 8)}
                    stats = {**stats, **get_stats(data, row["HOME_V"], row["AWAY_V"], 1, 4)}
                    stats = {**stats, **get_stats(data, row["HOME_V"], row["AWAY_V"], 2, 8)}
                    stats = {**stats, **get_stats(data, row["HOME_V"], row["AWAY_V"], 2, 4)}
            
            stats = {**stats, **get_stats(data, row["HOME_V"], row["AWAY_V"], 2)}
        
        if period == 3:
            if prev:
                stats = {**stats, **get_stats(data, row["HOME_V"], row["AWAY_V"], 1)}
                stats = {**stats, **get_stats(data, row["HOME_V"], row["AWAY_V"], 2)}

                if minutes:
                    stats = {**stats, **get_stats(data, row["HOME_V"], row["AWAY_V"], 1, 8)}
                    stats = {**stats, **get_stats(data, row["HOME_V"], row["AWAY_V"], 1, 4)}
                    stats = {**stats, **get_stats(data, row["HOME_V"], row["AWAY_V"], 2, 8)}
                    stats = {**stats, **get_stats(data, row["HOME_V"], row["AWAY_V"], 2, 4)}
                    stats = {**stats, **get_stats(data, row["HOME_V"], row["AWAY_V"], 3, 8)}
                    stats = {**stats, **get_stats(data, row["HOME_V"], row["AWAY_V"], 3, 4)}
            
            stats = {**stats, **get_stats(data, row["HOME_V"], row["AWAY_V"], 3)}

        for stat, val in stats.items():
            row[stat] = val

        return row
    
    games = []
    for idx, row in cur_df.iterrows():
        games.append(pd.DataFrame([fetch_game_stats(row)]))
    return games
