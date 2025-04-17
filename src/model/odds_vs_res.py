import pandas as pd
from datetime import date, datetime, timedelta
import os
from dotenv import load_dotenv
import urllib.request
from utils import NBA_TEAMS
import json
import numpy as np
from scipy.stats import norm
import click
load_dotenv()


'''data_df = pd.DataFrame({
    "ID": ["DET @ BOS", "IND @ BKN", "LAL @ MIA", "ORL @ PHI", "ATL @ MIL"],
    "models/winner.ubj": ["DET", "IND", "LAL", "ORL", "ATL"],
    "winner_prob": [0.5, 0.5, 0.5, 0.5, 0.5],
    "models/model.ubj": [200, 200, 200, 200, 200],
    "models/spread.ubj": [2, -2, 2, -2, 2],
    "models/homepoints.ubj": [100, 100, 100, 100, 100],
    "models/awaypoints.ubj": [100, 100, 100, 100, 100],
})

with open("message.txt", "r") as file:
    odds = json.loads(file.read())
'''
def compute_win_odds(se, pred, house):
    pred = abs(pred)
    house = abs(house)
    # Calculate the z-score for over
    z_over = (house - pred) / se
    probability_over = 1 - norm.cdf(z_over)  # Probability of hitting the over
    
    # Calculate the z-score for under
    z_under = (pred - house) / se
    probability_under = 1 - norm.cdf(z_under)  # Probability of hitting the under
    return round(100*probability_over, 2), round(100*probability_under,2)


@click.command
@click.option(
    "-a",
    "--algo",
    default="linear",
    type=click.Choice(
        ["xgb", "linear"], case_sensitive=False
    ),
)
@click.option(
    "-d",
    "--dataset",
    default="m",
    type=click.Choice(
        ["q", "q_prev", "m"], case_sensitive=False
    ),
)
@click.option(
    "-b",
    "--bookmaker",
    default="fanduel",
    type=click.Choice(
        ["draftkings", "fanduel"], case_sensitive=False
    ),
)
def main(algo, dataset, bookmaker):
    print(f"BEGINNING ODDS- {bookmaker}")
    if dataset == "q":
        dname = {
            1: "q1",
            2: "q2",
            3: "q3"
        }
    elif dataset == "q_prev":
        dname = {
            1: "q1",
            2: "q1_q2",
            3: "q1_q2_q3"
        }
    else:
        dname = {
            1: "q1_m",
            2: "q1_q2_m",
            3: "q1_q2_q3_m"
        }

    data_df = pd.read_parquet("livedata.parquet")

    limit_date = datetime.now().replace(hour=11, minute=59, second=0, microsecond=0)
    limit_date += timedelta(days=1)
    req = urllib.request.urlopen(f"https://api.the-odds-api.com/v4/sports/basketball_nba/events?apiKey={os.getenv("ODDS_API_KEY")}&commenceTimeTo={limit_date.isoformat()}Z")
    data = req.read()
    enc = req.info().get_content_charset('utf-8')
    events = json.loads(data.decode(enc))

    for idx, row in data_df.iterrows():
        long_away_team = NBA_TEAMS[row["ID"].split()[0]]
        for game in events:
            if game["away_team"] == long_away_team:
                game_id = game["id"]
                break
        else:
            print("Odds not found")
            break

        req = urllib.request.urlopen(f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{game_id}/odds?apiKey={os.getenv("ODDS_API_KEY")}&bookmakers={bookmaker}&markets=h2h,spreads,totals,team_totals&oddsFormat=american")
        data = req.read()
        enc = req.info().get_content_charset('utf-8')
        markets = json.loads(data.decode(enc))["bookmakers"][0]["markets"]


        for market in markets:
            if market["key"] == "h2h":
                h2h = (bookmaker, market["outcomes"][0]["price"], market["outcomes"][1]["price"])
            elif market["key"] == "spreads":
                spreads = (bookmaker, market["outcomes"][0]["price"], market["outcomes"][0]["point"], market["outcomes"][1]["price"], market["outcomes"][1]["point"])
            elif market["key"] == "totals":
                totals = (bookmaker, market["outcomes"][0]["price"], market["outcomes"][0]["point"], market["outcomes"][1]["price"], market["outcomes"][1]["point"])
            elif market["key"] == "team_totals":
                for x in market["outcomes"]:
                    if x["name"] == "Over":
                        if x["description"] == long_away_team:
                            away_over = (x["price"], x["point"])
                        else:
                            home_over = (x["price"], x["point"])
                    else:
                        if x["description"] == long_away_team:
                            away_under = (x["price"], x["point"])
                        else:
                            home_under = (x["price"], x["point"])

        book, homeprice, awayprice = h2h

        if homeprice < 0:
            homeprice = -homeprice/(-homeprice +100)
        else:
            homeprice = 100/ (homeprice +100)
        if awayprice < 0:
            awayprice = -awayprice/(-awayprice +100)
        else:
            awayprice = 100/(awayprice +100)

        h2h = (book, round(homeprice, 2), round(awayprice, 2))

        period = row["PERIOD"]
        se = []
        for name in ["total", "spread", "homepoints", "awaypoints"]:
            with open (f"stds/{algo}/{dname[period]}/{name}_std.json") as f:
                se.append(json.loads(f.read())["se"])   

        total_win_prob = compute_win_odds(se[0], row['total'],totals[2])
        spread_home_win_prob = compute_win_odds(se[1], row['spread'],spreads[2])
        spread_away_win_prob = compute_win_odds(se[1], row['spread'],spreads[4])
        home_win_prob = compute_win_odds(se[2], row['homepoints'],home_over[1])
        away_win_prob = compute_win_odds(se[3], row['awaypoints'],away_over[1])

        print(f"Bookmakers and Odds: {row["ID"]} : ending period {period}")
        print(f"  H2H: {h2h[0]} - Home Win Probability: {100*h2h[1]}% | Away Win Probability: {100*h2h[2]}%")
        print(f"  Spread: Home - {spreads[1]} (Point: {spreads[2]}) | Away - {spreads[3]} (Point: {spreads[4]})")
        print(f"  Total: Over - {totals[1]} (Point: {totals[2]}) | Under - {totals[3]} (Point: {totals[4]})")
        print(f"  Home: Over - {home_over[0]} (Point: {home_over[1]}) | Under - {home_under[0]} (Point: {home_under[1]})")
        print(f'Predictions:')
        print(f"  Winner: {row['winner']}")
        print(f"  Probability of win: {100*row['winner_prob']}%")
        print(f"  Second half winner: {row['second_winner']}")
        print(f"  Probability of second half win: {100*row['winner_prob2']}%")
        print(f"  Total points: {row['total']}")
        print(f"  Spread points: {row['spread']}")
        print(f"  Home points: {row['homepoints']}")
        print(f"  Away points: {row['awaypoints']}")
        print(f"Win Probability Calculation:")
        print(f"  Total Points Over Probability: {total_win_prob[0]}%")
        print(f"  Total Points Under Probability: {total_win_prob[1]}%")
        print(f"  Spread Winner Cover Probability: {spread_home_win_prob[0]}%")
        print(f"  Spread Loser Cover Probability: {spread_home_win_prob[1]}%")
        print(f"  Home Points Over Probability: {home_win_prob[0]}%")
        print(f"  Home Points Under Probability: {home_win_prob[1]}%")
        print(f"  Away Points Over Probability: {away_win_prob[0]}%")
        print(f"  Away Points Under Probability: {away_win_prob[1]}%")
        print("\n============================\n")


if __name__ == "__main__":
    main()