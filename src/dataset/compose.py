import pandas as pd
import os
import itertools

def pregame_features():
    base = ["GAME_ID", "HOME", "AWAY", "HFINAL", "AFINAL", "ODDS", "HREST", "AREST"]# , "HELO", "AELO"]
    base.extend([f"{t}ELO_Q{q}" for t in ['H', 'A'] for q in range(1,5)])
    base.extend([f"{t}{rating}" 
        for t in ["H", "A"]
        for rating in [
            "ORATING",
            "DRATING",
            "TCP",
            "APCT",
            "TOR",
            "AVG",
            "PACE_AVG",
            "PACE_AVG_Q4"
        ]])
    
    base.extend([
        f"{t}PIE_{n}"
        for t in ["H", "A"]
        for n in range(1,3)
    ])
    
    return base

def quarter_features(quarter):
    base = ["GAME_ID", "HOME", "AWAY", "HFINAL", "AFINAL", "ODDS", "HREST", "AREST"]# "HELO", "AELO"]
    base.extend([f"{t}ELO_Q{q}" for t in ['H', 'A'] for q in range(1,5)])
    base.extend([f"{t}{rating}" 
        for t in ["H", "A"]
        for rating in [
            "ORATING",
            "DRATING",
            "TCP",
            "APCT",
            "TOR",
            "AVG",
            "PACE_AVG",
            "PACE_AVG_Q4"
        ]])
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
        "PTS",
        "PACE"
    ]:
        base.extend([f"{t}{stat}_Q{quarter}" for t in ['H', 'A']])
    
    base.extend([
        f"{team}PIE_{i}_Q{quarter}"
        for team in ("H", "A")
        for i in range(1,3)
    ])

    base.extend([
        f"{team}PROJ_Q{quarter}"
        for team in ["H", "A"]
    ])
        
    base.extend([f"{m}_MARGIN_Q{quarter}" for m in ['MIN', 'MAX']])
    return base

def quarter_prev_features(quarter):
    base = ["GAME_ID", "HOME", "AWAY", "HFINAL", "AFINAL", "ODDS", "HREST", "AREST"]# , "HELO", "AELO"]
    #base = ["GAME_ID", "HFINAL", "AFINAL", "ODDS", "HREST", "AREST"]# , "HELO", "AELO"]
    #base.extend([f"{t}ELO_Q{q}" for t in ['H', 'A'] for q in range(1,5)])
    base.extend([f"{t}{rating}" 
        for t in ["H", "A"]
        for rating in [
            "ORATING",
            "DRATING",
            "TCP",
            "APCT",
            "TOR",
            "AVG",
            "PACE_AVG",
            "PACE_AVG_Q4",
        ]])
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
            "PTS",
            "PACE"
        ]:
            base.extend([f"{t}{stat}_Q{q}" for t in ['H', 'A']])
        
        base.extend([f"{m}_MARGIN_Q{q}" for m in ['MIN', 'MAX']])

        
    
    base.extend([
        f"{team}PROJ_Q{quarter}"
        for team in ["H", "A"]
    ])

    base.extend([
        f"{team}PIE_{i}_Q{quarter}"
        for team in ("H", "A")
        for i in range(1,3)
    ])

    base.extend([
        f"{t}PIE_{n}"
        for t in ["H", "A"]
        for n in range(1,3)
    ])

    return base

def quarter_m_features(quarter):
    base = ["GAME_ID", "HOME", "AWAY", "HFINAL", "AFINAL", "ODDS", "HREST", "AREST"]# , "HELO", "AELO"]
    base.extend([f"{t}ELO_Q{q}" for t in ['H', 'A'] for q in range(1,5)])
    base.extend([f"{t}{rating}" 
        for t in ["H", "A"]
        for rating in [
            "ORATING",
            "DRATING",
            "TCP",
            "APCT",
            "TOR",
            "AVG",
            "PACE_AVG",
            "PACE_AVG_Q4"
        ]])
    for q in range(1, quarter + 1):
        for suf in ["_M8", "_M4", ""]:
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
                base.extend([f"{t}{stat}_Q{q}{suf}" for t in ['H', 'A']])
            base.extend([f"{m}_MARGIN_Q{q}{suf}" for m in ['MIN', 'MAX']])

    for q in range(1, quarter + 1):
        for stat in [
            "PACE"
        ]:
            base.extend([f"{t}{stat}_Q{q}" for t in ['H', 'A']])

    base.extend([
        f"{team}PROJ_Q{quarter}"
        for team in ["H", "A"]
    ])

    base.extend([
        f"{team}PIE_{i}"
        for team in ("H", "A")
        for i in range(1,3)
    ])
    
    base.extend([
        f"{team}PIE_{i}_Q{quarter}"
        for team in ("H", "A")
        for i in range(1,3)
    ])
    
    return base

def optimal_features(quarter, prop):
    match quarter:
        case 'q1':
            match prop:
                case "homepoints":
                    return ['HOME', 'AWAY']
                case "awaypoints":
                    return ['HOME', 'AWAY']
                case "spread":
                    return ['HOME', 'AWAY']
                case "total":
                    return ['HOME', 'AWAY']
                case "winner":
                    return ['HOME', 'AWAY']
        case 'q2':
            match prop:
                case "homepoints":
                    return ['HOME', 'AWAY']
                case "awaypoints":
                    return ['HOME', 'AWAY']
                case "spread":
                    return ['HOME', 'AWAY']
                case "total":
                    return ['HOME', 'AWAY', 'ODDS', 'HREST', 'HELO', 'HELO_Q1', 'HELO_Q2', 'HELO_Q4', 'AELO_Q1', 'AELO_Q3', 'AELO_Q4', 'HPACE_AVG_Q4', 'ATCP', 'AAPCT', 'APACE_AVG', 'AFGM_Q1_M8', 'HFGA_Q1_M8', 'AFTM_Q1_M8', 'AFTA_Q1_M8', 'HTPA_Q1_M8', 'HOR_Q1_M8', 'AOR_Q1_M8', 'HDR_Q1_M8', 'HTR_Q1_M8', 'ATR_Q1_M8', 'HFO_Q1_M8', 'AFO_Q1_M8', 'HAS_Q1_M8', 'AAS_Q1_M8', 'HPTS_Q1_M8', 'MIN_MARGIN_Q1_M8', 'HFGM_Q1_M6', 'AFGM_Q1_M6', 'HFGA_Q1_M6', 'AFGA_Q1_M6', 'AFTA_Q1_M6', 'HTPM_Q1_M6', 'HTPA_Q1_M6', 'ATPA_Q1_M6', 'HOR_Q1_M6', 'HDR_Q1_M6', 'HTR_Q1_M6', 'HFO_Q1_M6', 'AFO_Q1_M6', 'HPTS_Q1_M6', 'APTS_Q1_M6', 'MIN_MARGIN_Q1_M6', 'HFGM_Q1_M4', 'AFGA_Q1_M4', 'HFTM_Q1_M4', 'AFTM_Q1_M4', 'AFTA_Q1_M4', 'HTPM_Q1_M4', 'HDR_Q1_M4', 'HTR_Q1_M4', 'HFO_Q1_M4', 'AFO_Q1_M4', 'HAS_Q1_M4', 'HPTS_Q1_M4', 'MIN_MARGIN_Q1_M4', 'MAX_MARGIN_Q1_M4', 'AFGM_Q1', 'HFGA_Q1', 'HTPM_Q1', 'ATPM_Q1', 'AOR_Q1', 'HDR_Q1', 'ADR_Q1', 'AFO_Q1', 'HAS_Q1', 'HPTS_Q1', 'MIN_MARGIN_Q1', 'MAX_MARGIN_Q1', 'HFGM_Q2_M8', 'AFGM_Q2_M8', 'HFGA_Q2_M8', 'AFTM_Q2_M8', 'HFTA_Q2_M8', 'HTPM_Q2_M8', 'ATPM_Q2_M8', 'HTPA_Q2_M8', 'ATPA_Q2_M8', 'HOR_Q2_M8', 'HDR_Q2_M8', 'ADR_Q2_M8', 'HTR_Q2_M8', 'ATR_Q2_M8', 'HAS_Q2_M8', 'AAS_Q2_M8', 'HPTS_Q2_M8', 'APTS_Q2_M8', 'MIN_MARGIN_Q2_M8', 'HFGM_Q2_M6', 'HFGA_Q2_M6', 'HFTM_Q2_M6', 'HFTA_Q2_M6', 'AFTA_Q2_M6', 'HTPM_Q2_M6', 'ATPA_Q2_M6', 'HOR_Q2_M6', 'HFO_Q2_M6', 'HAS_Q2_M6', 'AAS_Q2_M6', 'HPTS_Q2_M6', 'APTS_Q2_M6', 'MIN_MARGIN_Q2_M6', 'MAX_MARGIN_Q2_M6', 'HFGM_Q2_M4', 'AFGM_Q2_M4', 'AFTA_Q2_M4', 'ATPM_Q2_M4', 'HOR_Q2_M4', 'AOR_Q2_M4', 'ADR_Q2_M4', 'ATR_Q2_M4', 'AFO_Q2_M4', 'HAS_Q2_M4', 'AAS_Q2_M4', 'APTS_Q2_M4', 'MIN_MARGIN_Q2_M4', 'MAX_MARGIN_Q2_M4', 'HFGM_Q2', 'AFGM_Q2', 'HFGA_Q2', 'AFGA_Q2', 'HFTM_Q2', 'HFTA_Q2', 'AFTA_Q2', 'HTPM_Q2', 'HTPA_Q2', 'HOR_Q2', 'AOR_Q2', 'AFO_Q2', 'HAS_Q2', 'HPTS_Q2', 'APTS_Q2', 'MIN_MARGIN_Q2', 'HPACE_Q1', 'HPROJ_Q2', 'APROJ_Q2', 'HPIE_1', 'HPIE_3', 'HPIE_4', 'APIE_1', 'APIE_2', 'APIE_3', 'HPIE_1_Q2', 'HPIE_2_Q2', 'HPIE_3_Q2', 'HPIE_4_Q2', 'APIE_1_Q2', 'APIE_3_Q2', 'APIE_4_Q2', 'APIE_5_Q2']
                case "winner":
                    return ['HOME', 'AWAY']
        case 'q3':
            match prop:
                case "homepoints":
                    return ['HOME', 'AWAY']
                case "awaypoints":
                    return ['HOME', 'AWAY']
                case "spread":
                    return ['HOME', 'AWAY']
                case "total":
                    return ['HOME', 'AWAY']
                case "winner":
                    return ['HOME', 'AWAY']

def main():
    df = pd.read_parquet("datasets/live.parquet")

    def create_dataset(dataset_name, feature_list):
        dataset = df[feature_list]
        dataset.to_parquet(f"datasets/{dataset_name}.parquet", engine="pyarrow", compression="snappy")

    create_dataset("pregame", pregame_features())
    
    for name, q in zip(["q1", "q1_q2", "q1_q2_q3"], range(1,4)):
        create_dataset(name, quarter_prev_features(q))

    for name, q in zip(["q2", "q3"], range(2,4)):
        create_dataset(name, quarter_features(q))

    for name, q in zip(["q1_m", "q1_q2_m", "q1_q2_q3_m"], range(1,4)):
        create_dataset(name, quarter_m_features(q))
    
    def create_dataset(quarter, prop, feature_list):
        dataset = df[feature_list + ["GAME_ID", "HFINAL", "AFINAL"]]

        filename = f"datasets/{quarter}/{prop}.parquet"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        dataset.to_parquet(filename)

    for quarter, prop in itertools.product(['q1', 'q2', 'q3'], ['homepoints', 'awaypoints', 'spread', 'total', 'winner']):
        create_dataset(quarter, prop, optimal_features(quarter, prop))

if __name__ == "__main__":
    main()