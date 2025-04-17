import json

with open("stds/ratings.json", "r") as file:
    data = json.loads(file.read())

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

abbrev_map = {idx: abbrev for idx, abbrev in enumerate(nba_team_abbreviations)}


x = list(data.items())
x.sort(key=lambda x: x[1], reverse=True)


for y in x:
    print(f"{abbrev_map[int(y[0])]}: {y[1]}")