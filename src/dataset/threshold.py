import pandas as pd
import xgboost as xgb
from compose import quarter_m_features, quarter_prev_features
import matplotlib.pyplot as plt
import numpy as np
import os

BACKTEST_CUTOFF = 12500

def main():
    all_data = pd.read_parquet("datasets/live.parquet")

    all_data["spread"] = all_data["HFINAL"] - all_data["AFINAL"]
    all_data["total"] = all_data["HFINAL"] + all_data["AFINAL"]
    all_data["homepoints"] = all_data["HFINAL"]
    all_data["awaypoints"] = all_data["AFINAL"]
    all_data["winner"] = (all_data["HFINAL"] > all_data["AFINAL"]).astype(int)
    all_data["HOME"] = all_data["HOME"].astype("category")
    all_data["AWAY"] = all_data["AWAY"].astype("category")

    all_data = all_data[BACKTEST_CUTOFF:]

    odds_df = pd.read_parquet("datasets/odds.parquet")
    all_data = all_data.merge(odds_df, on="GAME_ID", how="inner")

    dname = {1: "q1", 2: "q1_q2", 3: "q1_q2_q3"}
    model_names = ["awaypoints", "homepoints", "spread", "total"]

    xgb_models = {
        mname: { 
            idx: create_regressor(f"models/{dname[idx]}/{mname}.ubj") 
            for idx in range(1,4)
        }
        for mname in model_names
    }

    winmodel = {
        idx: create_classifier(f"models/{dname[idx]}/winner.ubj") 
        for idx in range(1,4)
    }

    X = all_data[quarter_prev_features(2)].drop(["GAME_ID", "HFINAL", "AFINAL"], axis=1).sort_index(axis=1)

    for mname in model_names:
        all_data[f"{mname}_pred"] = xgb_models[mname][2].predict(X)

    all_data["winnerprob"] = winmodel[2].predict_proba(X)[:, 1]

    plot_thresholds(
        np.arange(0.1, 10, 0.01),
        12.,
        all_data["awaypoints_pred"], 
        all_data["AOVERUNDER_Q2"],
        all_data["AFINAL"] > all_data["AOVERUNDER_Q2"],
        all_data["AOVER_PRICE_Q2"],
        True,
        "thresholds/Q2/away_over.png"
    )

    plot_thresholds(
        np.arange(0.1, 10, 0.01), 
        8.,
        all_data["awaypoints_pred"], 
        all_data["AOVERUNDER_Q2"],
        all_data["AFINAL"] < all_data["AOVERUNDER_Q2"],
        all_data["AUNDER_PRICE_Q2"],
        False,
        "thresholds/Q2/away_under.png"
    )

    plot_thresholds(
        np.arange(0.1, 10, 0.01), 
        8,
        all_data["homepoints_pred"], 
        all_data["HOVERUNDER_Q2"],
        all_data["HFINAL"] > all_data["HOVERUNDER_Q2"],
        all_data["HOVER_PRICE_Q2"],
        True,
        "thresholds/Q2/home_over.png"
    )

    plot_thresholds(
        np.arange(0.1, 10, 0.01), 
        100,
        all_data["homepoints_pred"], 
        all_data["HOVERUNDER_Q2"],
        all_data["HFINAL"] < all_data["HOVERUNDER_Q2"],
        all_data["HUNDER_PRICE_Q2"],
        False,
        "thresholds/Q2/home_under.png"
    )

    plot_thresholds(
        np.arange(0.1, 15, 0.01), 
        8.5,
        all_data["total_pred"], 
        all_data["OVERUNDER_Q2"],
        all_data["total"] > all_data["OVERUNDER_Q2"],
        all_data["OVER_PRICE_Q2"],
        True,
        "thresholds/Q2/total_over.png"
    )

    plot_thresholds(
        np.arange(0.1, 15, 0.01), 
        10,
        all_data["total_pred"], 
        all_data["OVERUNDER_Q2"],
        all_data["total"] < all_data["OVERUNDER_Q2"],
        all_data["UNDER_PRICE_Q2"],
        False,
        "thresholds/Q2/total_under.png"
    )

    plot_thresholds(
        np.arange(0.01, 0.4, 0.002), 
        0.22,
        all_data["winnerprob"], 
        pd.Series(odds_to_implied(all_data["HH2H_Q2"].where(all_data["HH2H_Q2"] > 0, np.nan))),
        all_data["winner"],
        all_data["HH2H_Q2"],
        False,
        "thresholds/Q2/moneyline_home_plus.png"
    )

    plot_thresholds(
        np.arange(0.01, 0.4, 0.002), 
        100,
        all_data["winnerprob"], 
        pd.Series(odds_to_implied(all_data["HH2H_Q2"].where(all_data["HH2H_Q2"] < 0, np.nan))),
        all_data["winner"],
        all_data["HH2H_Q2"],
        False,
        "thresholds/Q2/moneyline_home_minus.png"
    )

    plot_thresholds(
        np.arange(0.01, 0.4, 0.002), 
        100,
        1 - all_data["winnerprob"], 
        pd.Series(odds_to_implied(all_data["AH2H_Q2"].where(all_data["AH2H_Q2"] > 0, np.nan))),
        1 - all_data["winner"],
        all_data["AH2H_Q2"],
        False,
        "thresholds/Q2/moneyline_away_plus.png"
    )
    
    plot_thresholds(
        np.arange(0.01, 0.4, 0.002), 
        0.3,
        1 - all_data["winnerprob"], 
        pd.Series(odds_to_implied(all_data["AH2H_Q2"].where(all_data["AH2H_Q2"] < 0, np.nan))),
        1 - all_data["winner"],
        all_data["AH2H_Q2"],
        False,
        "thresholds/Q2/moneyline_away_minus.png"
    )


def plot_thresholds(margins, cutoff, pred, vegas, target, prices, over, filename):
    bets_taken = []
    win_pct = []
    tot_winnings = []
    drawdown = []
    roi = []
    
    for margin in margins:
        winnings = pd.Series(np.where(
            ((pred - margin > vegas) & (pred - vegas < cutoff) if over 
            else (pred + margin < vegas) & (vegas - pred < cutoff)),
            np.where(
                target,
                winnings_on_bet(prices),
                -1
            ),
            np.nan
        )).dropna()
        
        bets_taken.append(winnings.count() / vegas.count() * 100) # uses vegas as total to handle missing data
        win_pct.append((winnings > 0).mean() * 100)
        tot_winnings.append(winnings.sum())
        drawdown.append((winnings.cumsum().cummax() - winnings.cumsum()).max())
        if winnings.count():
            roi.append((winnings.sum() + winnings.count()) / winnings.count() * 100)
        else:
            roi.append(0)
    
    _, axs = plt.subplots(3, 2, figsize=(12, 8))

    tot_winnings = pd.Series(tot_winnings)
    best_x = margins[tot_winnings.idxmax()]

    axs[0, 0].plot(margins, bets_taken, label="Bets Taken Pct", color='blue')
    axs[0, 0].set_title("Bets Taken")
    axs[0, 0].set_xlabel("Margin")
    axs[0, 0].set_ylabel("Taken %")

    most_bets = bets_taken[tot_winnings.idxmax()]
    axs[0, 0].axhline(y=most_bets, color='r', linestyle='--', label=f'Bets pct: {most_bets:.2f}')
    axs[0, 0].axvline(x=best_x, color='b', linestyle=':', label=f'At: {best_x:.2f}')
    axs[0, 0].legend()

    axs[0, 1].plot(margins, win_pct, label="Win Percentage", color='green')
    axs[0, 1].set_title("Win Percentage")
    axs[0, 1].set_xlabel("Margin")
    axs[0, 1].set_ylabel("Win %")

    best_pct = win_pct[tot_winnings.idxmax()]
    axs[0, 1].axhline(y=best_pct, color='r', linestyle='--', label=f'Win pct: {best_pct:.2f}')
    axs[0, 1].axvline(x=best_x, color='b', linestyle=':', label=f'At: {best_x:.2f}')
    axs[0, 1].legend()

    axs[1, 0].plot(margins, tot_winnings, label="Total Winnings", color='orange')
    axs[1, 0].set_title("Total Winnings")
    axs[1, 0].set_xlabel("Margin")
    axs[1, 0].set_ylabel("Winnings")

    global_max = tot_winnings.max()
    axs[1, 0].axhline(y=global_max, color='r', linestyle='--', label=f'Most won: {global_max:.2f}')
    axs[1, 0].axvline(x=best_x, color='b', linestyle=':', label=f'At: {best_x:.2f}')
    axs[1, 0].legend()

    axs[1, 1].plot(margins, drawdown, label="Drawdown", color='red')
    axs[1, 1].set_title("Max Drawdown")
    axs[1, 1].set_xlabel("Margin")
    axs[1, 1].set_ylabel("Drawdown")
    
    best_drawdown = drawdown[tot_winnings.idxmax()]
    axs[1, 1].axhline(y=best_drawdown, color='r', linestyle='--', label=f'Max drawdown: {best_drawdown:.2f}')
    axs[1, 1].axvline(x=best_x, color='b', linestyle=':', label=f'At: {best_x:.2f}')
    axs[1, 1].legend()

    axs[2, 0].plot(margins, roi, label="ROI %", color='green')
    axs[2, 0].set_title("ROI Percentage")
    axs[2, 0].set_xlabel("Margin")
    axs[2, 0].set_ylabel("ROI %")
    
    best_roi = roi[tot_winnings.idxmax()]
    axs[2, 0].axhline(y=best_roi, color='r', linestyle='--', label=f'Best roi: {best_roi:.2f}')
    axs[2, 0].axvline(x=best_x, color='b', linestyle=':', label=f'At: {best_x:.2f}')
    axs[2, 0].legend()

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.tight_layout()
    plt.savefig(filename)

def winnings_on_bet(odds):
    return np.where(
        odds > 0,
        odds / 100,
        100 / np.abs(odds),
    )

def odds_to_implied(odds):
    return np.where(
        odds > 0, 
        100 / (odds + 100), 
        np.abs(odds) / (np.abs(odds) + 100)
    )

def create_classifier(modelname):
    winmodel = xgb.XGBClassifier()
    winmodel.load_model(modelname)
    return winmodel

def create_regressor(name):
    model = xgb.XGBRegressor()
    model.load_model(name)
    return model

if __name__ == "__main__":
    main()