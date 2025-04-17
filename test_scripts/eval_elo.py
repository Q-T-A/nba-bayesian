import pandas as pd

df = pd.read_parquet("datasets/live.parquet")

games, correct = 0, 0

for idx, row in df[8000:].iterrows():
    games += 1
    correct += (row["HFINAL"] > row["AFINAL"]) == (row["HELO"] > row["AELO"])

print(correct / games)