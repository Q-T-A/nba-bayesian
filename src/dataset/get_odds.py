import numpy as np
import pandas as pd
from nba_api.stats.endpoints import boxscoresummaryv2
from datetime import datetime, timedelta, timezone
import time
from tqdm import tqdm
from dotenv import load_dotenv
import urllib
import json
import os
tqdm.pandas()
load_dotenv()

NBA_TEAMS = [
    "Atlanta Hawks",
    "Boston Celtics",
    "Brooklyn Nets",
    "Charlotte Hornets",
    "Chicago Bulls",
    "Cleveland Cavaliers",
    "Dallas Mavericks",
    "Denver Nuggets",
    "Detroit Pistons",
    "Golden State Warriors",
    "Houston Rockets",
    "Indiana Pacers",
    "Los Angeles Clippers",
    "Los Angeles Lakers",
    "Memphis Grizzlies",
    "Miami Heat",
    "Milwaukee Bucks",
    "Minnesota Timberwolves",
    "New Orleans Pelicans",
    "New York Knicks",
    "Oklahoma City Thunder",
    "Orlando Magic",
    "Philadelphia 76ers",
    "Phoenix Suns",
    "Portland Trail Blazers",
    "Sacramento Kings",
    "San Antonio Spurs",
    "Toronto Raptors",
    "Utah Jazz",
    "Washington Wizards"
]

def main(results_df):
    odds_df = pd.read_parquet("datasets/odds.parquet")

    new_ids = results_df[["GAME_ID", "HOME", "AWAY"]]
    latest_df = new_ids[~new_ids["GAME_ID"].isin(odds_df["GAME_ID"])]

    def get_odds(row):
        # Fetch game date
        if int(row["GAME_ID"][3:5]) < 23:
            return row
        try:
            game_date = boxscoresummaryv2.BoxScoreSummaryV2(row["GAME_ID"]).get_data_frames()[0].iloc[0]["GAME_DATE_EST"]
        except:
            print("FAILED", row["GAME_ID"])
            time.sleep(2)
            try:
                game_date = boxscoresummaryv2.BoxScoreSummaryV2(row["GAME_ID"]).get_data_frames()[0].iloc[0]["GAME_DATE_EST"]
            except:
                return row
        
        # Fetch games at date
        game_date = datetime.fromisoformat(game_date)
        game_date = game_date.replace(hour=16, minute=0, second=0)

        home_team = NBA_TEAMS[row["HOME"]]

        req = urllib.request.urlopen(f"https://api.the-odds-api.com/v4/historical/sports/basketball_nba/events?apiKey={os.getenv('ODDS_API_KEY')}&date={game_date.isoformat()}Z")
        data = req.read()
        enc = req.info().get_content_charset('utf-8')
        events = json.loads(data.decode(enc))
        time.sleep(0.2)

        for event in events["data"]:
            if event["home_team"] == home_team:
                game_id = event["id"]
                commence_time = event["commence_time"]
                break
        else:
            print("EVENT NOT FOUND AT", row["GAME_ID"])
            return row

        pbp = pd.read_parquet(f"pbps/{row['GAME_ID']}.parquet")
        for idx, pbprow in pbp.iterrows():
            if pbprow["EVENTMSGTYPE"] == 13 and pbprow["PERIOD"] == 1:
                p1_end = pbprow["WCTIMESTRING"].split(" ")
                p1_end_time = game_date
                if p1_end[1] == "PM":
                    p1_end = p1_end[0].split(":")
                    if p1_end[0] == "12":
                        p1_end[0] = 0
                    p1_end_time = p1_end_time.replace(hour=int(p1_end[0]) + 12, minute=int(p1_end[1]))
                else:
                    p1_end = p1_end[0].split(":")
                    p1_end_time += timedelta(days=1)
                    if p1_end[0] == "12":
                        p1_end[0] = 0
                    p1_end_time = p1_end_time.replace(hour=int(p1_end[0]), minute=int(p1_end[1]))

            elif pbprow["EVENTMSGTYPE"] == 13 and pbprow["PERIOD"] == 2:
                p2_end = pbprow["WCTIMESTRING"].split(" ")
                p2_end_time = game_date
                if p2_end[1] == "PM":
                    p2_end = p2_end[0].split(":")
                    if p2_end[0] == "12":
                        p2_end[0] = 0
                    p2_end_time = p2_end_time.replace(hour=int(p2_end[0]) + 12, minute=int(p2_end[1]))
                else:
                    p2_end = p2_end[0].split(":")
                    p2_end_time += timedelta(days=1)
                    if p2_end[0] == "12":
                        p2_end[0] = 0
                    p2_end_time = p2_end_time.replace(hour=int(p2_end[0]), minute=int(p2_end[1]))

            elif pbprow["EVENTMSGTYPE"] == 13 and pbprow["PERIOD"] == 3:
                p3_end = pbprow["WCTIMESTRING"].split(" ")
                p3_end_time = game_date
                if p3_end[1] == "PM":
                    p3_end = p3_end[0].split(":")
                    if p3_end[0] == "12":
                        p3_end[0] = 0
                    p3_end_time = p3_end_time.replace(hour=int(p3_end[0]) + 12, minute=int(p3_end[1]))
                else:
                    p3_end = p3_end[0].split(":")
                    p3_end_time += timedelta(days=1)
                    if p3_end[0] == "12":
                        p3_end[0] = 0
                    p3_end_time = p3_end_time.replace(hour=int(p3_end[0]), minute=int(p3_end[1]))
                break
        
        start_time = datetime.strptime(commence_time, "%Y-%m-%dT%H:%M:%SZ")
        start_time -= timedelta(minutes=15)

        odds_cutoff = datetime(2022, 9, 1)

        if game_date >= odds_cutoff: 
            p1_end_time += timedelta(hours=5, minutes=4)
            p2_end_time += timedelta(hours=5, minutes=10)
            p3_end_time += timedelta(hours=5, minutes=4)
        else:
            p1_end_time += timedelta(hours=5, minutes=6, seconds=30)
            p2_end_time += timedelta(hours=5, minutes=5)
            p3_end_time += timedelta(hours=5, minutes=6, seconds=30)

        bookmaker = "draftkings"

        try:
            req = urllib.request.urlopen(f"https://api.the-odds-api.com/v4/historical/sports/basketball_nba/events/{game_id}/odds?apiKey={os.getenv("ODDS_API_KEY")}&bookmakers={bookmaker}&markets=h2h,spreads,totals,team_totals&oddsFormat=american&date={start_time.isoformat()}Z")
        except:
            print(row["GAME_ID"], "not found start")
            return row
        data = req.read()
        enc = req.info().get_content_charset('utf-8')
        markets = json.loads(data.decode(enc))["data"]["bookmakers"][0]["markets"]
        time.sleep(0.2)

        for market in markets:
            if market["key"] == "h2h":
                for outcome in market["outcomes"]:
                    if outcome["name"] == home_team:
                        row["HH2H_P"] = outcome["price"]
                    else:
                        row["AH2H_P"] = outcome["price"]
            elif market["key"] == "spreads":
                for outcome in market["outcomes"]:
                    if outcome["name"] == home_team:
                        row["SPREAD_P"] = outcome["point"]
                        row["HSPREAD_PRICE_P"] = outcome["price"]
                    else:
                        row["ASPREAD_PRICE_P"] = outcome["price"]
            elif market["key"] == "totals":
                for outcome in market["outcomes"]:
                    if outcome["name"] == "Over":
                        row["OVERUNDER_P"] = outcome["point"]
                        row["OVER_PRICE_P"] = outcome["price"]
                    else:
                        row["UNDER_PRICE_P"] = outcome["price"]
            elif market["key"] == "team_totals":
                for outcome in market["outcomes"]:
                    if outcome["name"] == "Over":
                        if outcome["description"] == home_team:
                            row["HOVERUNDER_P"] = outcome["point"]
                            row["HOVER_PRICE_P"] = outcome["price"]
                        else:
                            row["AOVERUNDER_P"] = outcome["point"]
                            row["AOVER_PRICE_P"] = outcome["price"]
                    else:
                        if outcome["description"] == home_team:
                            row["HUNDER_PRICE_P"] = outcome["price"]
                        else:
                            row["AUNDER_PRICE_P"] = outcome["price"]

        try:
            req = urllib.request.urlopen(f"https://api.the-odds-api.com/v4/historical/sports/basketball_nba/events/{game_id}/odds?apiKey={os.getenv("ODDS_API_KEY")}&bookmakers={bookmaker}&markets=h2h,spreads,totals,team_totals&oddsFormat=american&date={p1_end_time.isoformat()}Z")
        except:
            print(row["GAME_ID"], "not found q1")
            return row
        data = req.read()
        enc = req.info().get_content_charset('utf-8')
        markets = json.loads(data.decode(enc))["data"]["bookmakers"][0]["markets"]
        time.sleep(0.2)

        for market in markets:
            if market["key"] == "h2h":
                for outcome in market["outcomes"]:
                    if outcome["name"] == home_team:
                        row["HH2H_Q1"] = outcome["price"]
                    else:
                        row["AH2H_Q1"] = outcome["price"]
            elif market["key"] == "spreads":
                for outcome in market["outcomes"]:
                    if outcome["name"] == home_team:
                        row["SPREAD_Q1"] = outcome["point"]
                        row["HSPREAD_PRICE_Q1"] = outcome["price"]
                    else:
                        row["ASPREAD_PRICE_Q1"] = outcome["price"]
            elif market["key"] == "totals":
                for outcome in market["outcomes"]:
                    if outcome["name"] == "Over":
                        row["OVERUNDER_Q1"] = outcome["point"]
                        row["OVER_PRICE_Q1"] = outcome["price"]
                    else:
                        row["UNDER_PRICE_Q1"] = outcome["price"]
            elif market["key"] == "team_totals":
                for outcome in market["outcomes"]:
                    if outcome["name"] == "Over":
                        if outcome["description"] == home_team:
                            row["HOVERUNDER_Q1"] = outcome["point"]
                            row["HOVER_PRICE_Q1"] = outcome["price"]
                        else:
                            row["AOVERUNDER_Q1"] = outcome["point"]
                            row["AOVER_PRICE_Q1"] = outcome["price"]
                    else:
                        if outcome["description"] == home_team:
                            row["HUNDER_PRICE_Q1"] = outcome["price"]
                        else:
                            row["AUNDER_PRICE_Q1"] = outcome["price"]

        try:
            req = urllib.request.urlopen(f"https://api.the-odds-api.com/v4/historical/sports/basketball_nba/events/{game_id}/odds?apiKey={os.getenv("ODDS_API_KEY")}&bookmakers={bookmaker}&markets=h2h,spreads,totals,team_totals&oddsFormat=american&date={p2_end_time.isoformat()}Z")
        except:
            try:
                p2_end_time -= timedelta(minutes=5)
                req = urllib.request.urlopen(f"https://api.the-odds-api.com/v4/historical/sports/basketball_nba/events/{game_id}/odds?apiKey={os.getenv("ODDS_API_KEY")}&bookmakers={bookmaker}&markets=h2h,spreads,totals,team_totals&oddsFormat=american&date={p2_end_time.isoformat()}Z")
            except:
                print(row["GAME_ID"], "ht stats not found")
                return row
        data = req.read()
        enc = req.info().get_content_charset('utf-8')
        try:
            markets = json.loads(data.decode(enc))["data"]["bookmakers"][0]["markets"]
        except:
            return row
        time.sleep(0.2)

        for market in markets:
            if market["key"] == "h2h":
                for outcome in market["outcomes"]:
                    if outcome["name"] == home_team:
                        row["HH2H_Q2"] = outcome["price"]
                    else:
                        row["AH2H_Q2"] = outcome["price"]
            elif market["key"] == "spreads":
                for outcome in market["outcomes"]:
                    if outcome["name"] == home_team:
                        row["SPREAD_Q2"] = outcome["point"]
                        row["HSPREAD_PRICE_Q2"] = outcome["price"]
                    else:
                        row["ASPREAD_PRICE_Q2"] = outcome["price"]
            elif market["key"] == "totals":
                for outcome in market["outcomes"]:
                    if outcome["name"] == "Over":
                        row["OVERUNDER_Q2"] = outcome["point"]
                        row["OVER_PRICE_Q2"] = outcome["price"]
                    else:
                        row["UNDER_PRICE_Q2"] = outcome["price"]
            elif market["key"] == "team_totals":
                for outcome in market["outcomes"]:
                    if outcome["name"] == "Over":
                        if outcome["description"] == home_team:
                            row["HOVERUNDER_Q2"] = outcome["point"]
                            row["HOVER_PRICE_Q2"] = outcome["price"]
                        else:
                            row["AOVERUNDER_Q2"] = outcome["point"]
                            row["AOVER_PRICE_Q2"] = outcome["price"]
                    else:
                        if outcome["description"] == home_team:
                            row["HUNDER_PRICE_Q2"] = outcome["price"]
                        else:
                            row["AUNDER_PRICE_Q2"] = outcome["price"]

        try:
            req = urllib.request.urlopen(f"https://api.the-odds-api.com/v4/historical/sports/basketball_nba/events/{game_id}/odds?apiKey={os.getenv("ODDS_API_KEY")}&bookmakers={bookmaker}&markets=h2h,spreads,totals,team_totals&oddsFormat=american&date={p3_end_time.isoformat()}Z")
        except:
            print(row["GAME_ID"], "Q3 odds not found")
            return row
        data = req.read()
        enc = req.info().get_content_charset('utf-8')
        try:
            markets = json.loads(data.decode(enc))["data"]["bookmakers"][0]["markets"]
        except:
            return row
        time.sleep(0.2)

        for market in markets:
            if market["key"] == "h2h":
                for outcome in market["outcomes"]:
                    if outcome["name"] == home_team:
                        row["HH2H_Q3"] = outcome["price"]
                    else:
                        row["AH2H_Q3"] = outcome["price"]
            elif market["key"] == "spreads":
                for outcome in market["outcomes"]:
                    if outcome["name"] == home_team:
                        row["SPREAD_Q3"] = outcome["point"]
                        row["HSPREAD_PRICE_Q3"] = outcome["price"]
                    else:
                        row["ASPREAD_PRICE_Q3"] = outcome["price"]
            elif market["key"] == "totals":
                for outcome in market["outcomes"]:
                    if outcome["name"] == "Over":
                        row["OVERUNDER_Q3"] = outcome["point"]
                        row["OVER_PRICE_Q3"] = outcome["price"]
                    else:
                        row["UNDER_PRICE_Q3"] = outcome["price"]
            elif market["key"] == "team_totals":
                for outcome in market["outcomes"]:
                    if outcome["name"] == "Over":
                        if outcome["description"] == home_team:
                            row["HOVERUNDER_Q3"] = outcome["point"]
                            row["HOVER_PRICE_Q3"] = outcome["price"]
                        else:
                            row["AOVERUNDER_Q3"] = outcome["point"]
                            row["AOVER_PRICE_Q3"] = outcome["price"]
                    else:
                        if outcome["description"] == home_team:
                            row["HUNDER_PRICE_Q3"] = outcome["price"]
                        else:
                            row["AUNDER_PRICE_Q3"] = outcome["price"]

        return row

    print("Get odds")
    latest_df = latest_df.progress_apply(get_odds, axis=1)
    
    new_odds = pd.concat([odds_df, latest_df.drop(["HOME", "AWAY"], axis=1)])
    #print(new_odds)

    return new_odds


if __name__ == "__main__":
    results_df = pd.read_parquet("datasets/results.parquet")
    odds_df = main(results_df) 
    odds_df.to_parquet("datasets/odds.parquet", engine="pyarrow", compression="snappy")