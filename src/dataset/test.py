from nba_api.live.nba.endpoints import scoreboard
import json

x = scoreboard.ScoreBoard()

with open("test.json", "w") as file:
    file.write(json.dumps(x.games.get_dict(), indent=4))