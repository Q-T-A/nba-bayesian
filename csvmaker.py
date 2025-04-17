import pandas as pd
df = pd.read_parquet("datasets/q1_m.parquet")
df.to_csv("q1_m.csv")