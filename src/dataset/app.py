from flask import Flask, render_template, jsonify
from nba_api.live.nba.endpoints import scoreboard
import subprocess
from floor import floor
import threading
import time

app = Flask(__name__)

def get_todays_games():
    todays_data = scoreboard.ScoreBoard().games.get_dict()
    # with open("test.json", "r") as file:
    #     todays_data = json.loads(file.read())

    game_data = [{
        "GAME_ID": game["gameId"],
        "HOME_S": game["homeTeam"]["teamTricode"],
        "HOME_V": f"{game["homeTeam"]["teamCity"]} {game["homeTeam"]["teamName"]}",
        "AWAY_S": game["awayTeam"]["teamTricode"],
        "AWAY_V": f"{game["awayTeam"]["teamCity"]} {game["awayTeam"]["teamName"]}",
        "SCORE": f"{game["awayTeam"]["score"]} - {game["homeTeam"]["score"]} ({'FINAL' if game['gameStatusText'] == 'Final' else f'Starts at {game['gameStatusText']}' if game['gameStatus'] == 1 else f'{game['gameClock']} {game['period']}Q'})",
        "IS_END": game["gameStatus"] in { 2, 7 } and game["gameClock"] == "PT00M00.00S"
        }
        for game in todays_data
    ]

    return game_data

games = get_todays_games()

def get_game_prediction(game_id):
    return {
        "ID": 0,
        "winner": 1,
        "winnerprob": 0.78,
        "total": 200,
        "spread": 5,
        "homepoints": 100,
        "awaypoints": 100,

        "total_conf": { "56%": { "floor": -1, "ceiling": 1}, "60%": { "floor": -2, "ceiling": 2}, "65%": { "floor": -3, "ceiling": 3}, "70%": { "floor": -4, "ceiling": 4} },
        "spread_conf": { "56%": { "floor": -1, "ceiling": 1}, "60%": { "floor": -2, "ceiling": 2}, "65%": { "floor": -3, "ceiling": 3}, "70%": { "floor": -4, "ceiling": 4} },
        "home_conf": { "56%": { "floor": -1, "ceiling": 1}, "60%": { "floor": -2, "ceiling": 2}, "65%": { "floor": -3, "ceiling": 3}, "70%": { "floor": -4, "ceiling": 4} },
        "away_conf": { "56%": { "floor": -1, "ceiling": 1}, "60%": { "floor": -2, "ceiling": 2}, "65%": { "floor": -3, "ceiling": 3}, "70%": { "floor": -4, "ceiling": 4} },

    }

updated_games = {game["GAME_ID"]: game for game in games}

def update_games():
    while True:
        new_games = get_todays_games()
        for game in new_games:
            if game["IS_END"] and not updated_games[game["GAME_ID"]]["IS_END"]:
                message = f"Game {game['AWAY_V']} at {game['HOME_V']} END, {game['SCORE']}"
                script = f'display notification "{message}" with title "arbitage-tech" & do shell script "afplay /System/Library/Sounds/Glass.aiff"'
                subprocess.run(["osascript", "-e", script])
                preds = floor("xgb", "m")
                updated_games[game["GAME_ID"]].update(preds[game["GAME_ID"]])
            if not game["IS_END"] and updated_games[game["GAME_ID"]]["IS_END"]:
                updated_games[game["GAME_ID"]] = game
            else:
                updated_games[game["GAME_ID"]].update(game)
        time.sleep(4)

threading.Thread(target=update_games, daemon=True).start()

@app.route('/')
def index():
    return render_template('index.html', games=games)

@app.route('/scores')
def scores():
    return jsonify(updated_games)

if __name__ == '__main__':
    app.run(debug=True)
    # app.run()
