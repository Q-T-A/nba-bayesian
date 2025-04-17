import pandas as pd
import xgboost as xgb
import math
import matplotlib.pyplot as plt

def create_classifier(modelname):
        winmodel = xgb.XGBClassifier()
        winmodel.load_model(modelname)
        return winmodel

def implied_pct(american_odds):

    if american_odds > 0:
        return (100 / (american_odds + 100)) * 100
    elif american_odds < 0:
        return (abs(american_odds) / (abs(american_odds) + 100)) * 100
    else:
        raise Exception

def quarter_prev_features(quarter):
        base = ["HOME", "AWAY", "ODDS", "HREST", "AREST"]# , "HELO", "AELO"]
        base.extend([f"{t}ELO_Q{q}" for t in ['H', 'A'] for q in range(1,5)])
        base.extend([f"{t}{rating}" 
            for t in ["H", "A"]
            for rating in [
                "ORATING",
                "DRATING",
                "TCP",
                "APCT",
                "TOR",
            ]])
        if quarter == 0:
            base.extend([
            f"{t}PIE_{n}"
            for t in ["H", "A"]
            for n in range(1,6)
        ])
        else:
            for q in range(1, quarter + 1):
                for stat in [
                    "FGM", 
                    "FGA", 
                    "FTM", 
                    "FTA",
                    "TPM",
                    "TPA",
                    "TR",
                    "FO",
                    "AS",
                    "PTS"
                ]:
                    base.extend([f"{t}{stat}_Q{q}" for t in ['H', 'A']])
                
                base.extend([f"{m}_MARGIN_Q{q}" for m in ['MIN', 'MAX']])
            
            base.extend([
                f"{team}PIE_{i}_Q{quarter}"
                for team in ("H", "A")
                for i in range(1,6)
            ])

            # base.extend([
            #     f"{t}PIE_{n}"
            #     for t in ["H", "A"]
            #     for n in range(1,6)
            # ])

        return base

data = pd.read_parquet("datasets/live.parquet")
data = data[-700:]

dname = {0: "pregame", 1: "q1", 2: "q1_q2", 3: "q1_q2_q3"}

winmodel = {
    idx: create_classifier(f"models/{dname[idx]}/winner.ubj") for idx in range(4)
}

scatter_x = [[[] for _ in range(8)] for _ in range(4)]
scatter_y = [[[] for _ in range(8)] for _ in range(4)]

for idx, row in data.iterrows():
    for i, ext in enumerate(['P', 'Q1', 'Q2']):
        X = pd.DataFrame([row])
        X = X[quarter_prev_features(i)]
        X["HOME"] = X["HOME"].astype("category")
        X["AWAY"] = X["AWAY"].astype("category")
        X = X.sort_index(axis=1)

        winner = winmodel[i].predict(X)[0]
        home_prob = winmodel[i].predict_proba(X)[0][1]
        
        pred_winner = row[f"HH2H_{ext}"] < row[f"AH2H_{ext}"]

        if math.isnan(row[f"HH2H_{ext}"]) or math.isnan(row[f"AH2H_{ext}"]):
            continue

        if row["HFINAL"] > row["AFINAL"]:
            if pred_winner:
                if winner:
                    scatter_x[i][0].append(home_prob * 100)
                    scatter_y[i][0].append(implied_pct(row[f"HH2H_{ext}"]))
                else:
                    scatter_x[i][1].append(home_prob * 100)
                    scatter_y[i][1].append(implied_pct(row[f"HH2H_{ext}"]))
            else:
                if winner:
                    scatter_x[i][2].append(home_prob * 100)
                    scatter_y[i][2].append(implied_pct(row[f"HH2H_{ext}"]))
                else:
                    scatter_x[i][3].append(home_prob * 100)
                    scatter_y[i][3].append(implied_pct(row[f"HH2H_{ext}"]))
        else:
            if pred_winner:
                if winner:
                    scatter_x[i][4].append(home_prob * 100)
                    scatter_y[i][4].append(implied_pct(row[f"HH2H_{ext}"]))
                else:
                    scatter_x[i][5].append(home_prob * 100)
                    scatter_y[i][5].append(implied_pct(row[f"HH2H_{ext}"]))
            else:
                if winner:
                    scatter_x[i][6].append(home_prob * 100)
                    scatter_y[i][6].append(implied_pct(row[f"HH2H_{ext}"]))
                else:
                    scatter_x[i][7].append(home_prob * 100)
                    scatter_y[i][7].append(implied_pct(row[f"HH2H_{ext}"]))

thresh = 6

for idx, ext in enumerate(['P', 'Q1', 'Q2']):
    plt.figure(figsize=(10, 6))
    plt.scatter(scatter_x[idx][0], scatter_y[idx][0], color="blue", linewidth=2)
    plt.xlabel("Our pred")
    plt.ylabel("Vegas implied prob")
    plt.xlim(0,100)
    plt.ylim(0,100)
    plt.plot([0, 100], [0, 100], color="red", linestyle="--")
    plt.plot([thresh, 100], [0, 100 - thresh], color="orange", linestyle="--")
    plt.plot([0, 100 - thresh], [thresh, 100], color="orange", linestyle="--")
    plt.axhline(y=50, color="purple", linestyle=":")
    plt.axvline(x=50, color="purple", linestyle=":")
    plt.title("Winner H, Ours H, Vegas H")
    plt.legend()
    plt.savefig(f"gout/{ext}/Probs_0.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(scatter_x[idx][1], scatter_y[idx][1], color="blue", linewidth=2)
    plt.xlabel("Our pred")
    plt.ylabel("Vegas implied prob")
    plt.xlim(0,100)
    plt.ylim(0,100)
    plt.plot([0, 100], [0, 100], color="red", linestyle="--")
    plt.plot([thresh, 100], [0, 100 - thresh], color="orange", linestyle="--")
    plt.plot([0, 100 - thresh], [thresh, 100], color="orange", linestyle="--")
    plt.axhline(y=50, color="purple", linestyle=":")
    plt.axvline(x=50, color="purple", linestyle=":")
    plt.title("Winner H, Ours A, Vegas H")
    plt.legend()
    plt.savefig(f"gout/{ext}/Probs_1.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(scatter_x[idx][2], scatter_y[idx][2], color="blue", linewidth=2)
    plt.xlabel("Our pred")
    plt.ylabel("Vegas implied prob")
    plt.xlim(0,100)
    plt.ylim(0,100)
    plt.plot([0, 100], [0, 100], color="red", linestyle="--")
    plt.plot([thresh, 100], [0, 100 - thresh], color="orange", linestyle="--")
    plt.plot([0, 100 - thresh], [thresh, 100], color="orange", linestyle="--")
    plt.axhline(y=50, color="purple", linestyle=":")
    plt.axvline(x=50, color="purple", linestyle=":")
    plt.title("Winner H, Ours H, Vegas A")
    plt.legend()
    plt.savefig(f"gout/{ext}/Probs_2.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(scatter_x[idx][3], scatter_y[idx][3], color="blue", linewidth=2)
    plt.xlabel("Our pred")
    plt.ylabel("Vegas implied prob")
    plt.xlim(0,100)
    plt.ylim(0,100)
    plt.plot([0, 100], [0, 100], color="red", linestyle="--")
    plt.plot([thresh, 100], [0, 100 - thresh], color="orange", linestyle="--")
    plt.plot([0, 100 - thresh], [thresh, 100], color="orange", linestyle="--")
    plt.axhline(y=50, color="purple", linestyle=":")
    plt.axvline(x=50, color="purple", linestyle=":")
    plt.title("Winner H, Ours A, Vegas A")
    plt.legend()
    plt.savefig(f"gout/{ext}/Probs_3.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(scatter_x[idx][4], scatter_y[idx][4], color="blue", linewidth=2)
    plt.xlabel("Our pred")
    plt.ylabel("Vegas implied prob")
    plt.xlim(0,100)
    plt.ylim(0,100)
    plt.plot([0, 100], [0, 100], color="red", linestyle="--")
    plt.plot([thresh, 100], [0, 100 - thresh], color="orange", linestyle="--")
    plt.plot([0, 100 - thresh], [thresh, 100], color="orange", linestyle="--")
    plt.axhline(y=50, color="purple", linestyle=":")
    plt.axvline(x=50, color="purple", linestyle=":")
    plt.title("Winner A, Ours H, Vegas H")
    plt.legend()
    plt.savefig(f"gout/{ext}/Probs_4.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(scatter_x[idx][5], scatter_y[idx][5], color="blue", linewidth=2)
    plt.xlabel("Our pred")
    plt.ylabel("Vegas implied prob")
    plt.xlim(0,100)
    plt.ylim(0,100)
    plt.plot([0, 100], [0, 100], color="red", linestyle="--")
    plt.plot([thresh, 100], [0, 100 - thresh], color="orange", linestyle="--")
    plt.plot([0, 100 - thresh], [thresh, 100], color="orange", linestyle="--")
    plt.axhline(y=50, color="purple", linestyle=":")
    plt.axvline(x=50, color="purple", linestyle=":")
    plt.title("Winner A, Ours A, Vegas H")
    plt.legend()
    plt.savefig(f"gout/{ext}/Probs_5.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(scatter_x[idx][6], scatter_y[idx][6], color="blue", linewidth=2)
    plt.xlabel("Our pred")
    plt.ylabel("Vegas implied prob")
    plt.xlim(0,100)
    plt.ylim(0,100)
    plt.plot([0, 100], [0, 100], color="red", linestyle="--")
    plt.plot([thresh, 100], [0, 100 - thresh], color="orange", linestyle="--")
    plt.plot([0, 100 - thresh], [thresh, 100], color="orange", linestyle="--")
    plt.axhline(y=50, color="purple", linestyle=":")
    plt.axvline(x=50, color="purple", linestyle=":")
    plt.title("Winner A, Ours H, Vegas A")
    plt.legend()
    plt.savefig(f"gout/{ext}/Probs_6.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(scatter_x[idx][7], scatter_y[idx][7], color="blue", linewidth=2)
    plt.xlabel("Our pred")
    plt.ylabel("Vegas implied prob")
    plt.xlim(0,100)
    plt.ylim(0,100)
    plt.plot([0, 100], [0, 100], color="red", linestyle="--")
    plt.plot([thresh, 100], [0, 100 - thresh], color="orange", linestyle="--")
    plt.plot([0, 100 - thresh], [thresh, 100], color="orange", linestyle="--")
    plt.axhline(y=50, color="purple", linestyle=":")
    plt.axvline(x=50, color="purple", linestyle=":")
    plt.title("Winner A, Ours A, Vegas A")
    plt.legend()
    plt.savefig(f"gout/{ext}/Probs_7.png")
    plt.close()
