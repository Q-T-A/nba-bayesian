import numpy as np
import pandas as pd
import random
import multiprocessing
from deap import base, creator, tools, algorithms
import xgboost as xgb

BACKTEST_CUTOFF = 12500

def winnings_on_bet(odds):
    return np.where(
        odds > 0,
        odds / 100,
        100 / np.abs(odds),
    )

def all_features(quarter):
    base = ["HOME", "AWAY", "ODDS", "HREST", "AREST", "HELO", "AELO"]
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
        for suf in ["_M8", "_M6", "_M4", ""]:
            for stat in [
                "FGM", 
                "FGA", 
                "FTM", 
                "FTA",
                "TPM",
                "TPA",
                "OR",
                "DR",
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
        for i in range(1,6)
    ])
    
    base.extend([
        f"{team}PIE_{i}_Q{quarter}"
        for team in ("H", "A")
        for i in range(1,6)
    ])
    
    return base

pred_features = ["UNDER_PRICE_Q2", "OVERUNDER_Q2"]

all_data = pd.read_parquet("datasets/live.parquet")
odds_data = pd.read_parquet("datasets/odds.parquet")
all_data = pd.concat([all_data, odds_data], axis=1)
all_data["total"] = all_data["HFINAL"] + all_data["AFINAL"]
all_data["HOME"] = all_data["HOME"].astype("category")
all_data["AWAY"] = all_data["AWAY"].astype("category")

X_train = all_data.loc[:BACKTEST_CUTOFF, all_features(2)]
y_train = all_data.loc[:BACKTEST_CUTOFF, "total"]

X_test = all_data.loc[BACKTEST_CUTOFF:, all_features(2) + pred_features]
y_test = all_data.loc[BACKTEST_CUTOFF:, "total"]

# Define Fitness and Individual
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def init_individual():
    return [random.randint(0, 1) for _ in range(X_train.shape[1])]

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, init_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    selected_features = np.where(np.array(individual) == 1)[0]
    if len(selected_features) == 0:
        return (0,)

    model = xgb.XGBRegressor(
        max_depth=4, 
        learning_rate=0.01, 
        n_estimators=500,
        subsample=0.8, 
        colsample_bytree=0.8, 
        random_state=42,
        objective="reg:squarederror", 
        enable_categorical=True,
        # tree_method="gpu_hist",
    )
    model.fit(X_train.iloc[:, selected_features], y_train)
    y_pred = model.predict(X_test.drop(pred_features, axis=1).iloc[:, selected_features])

    winnings = np.nansum(np.where(
        (y_pred + 4.5 < X_test["OVERUNDER_Q2"])
        & (X_test["OVERUNDER_Q2"] - y_pred < 10),
        np.where(
            y_test < X_test["OVERUNDER_Q2"],
            winnings_on_bet(X_test["UNDER_PRICE_Q2"]),
            -1
        ),
        np.nan
    ))

    return (winnings,)

# Register Genetic Operators
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

toolbox.register("map", multiprocessing.get_context("fork").Pool().map) # mac thing?

def run_ga(n_generations, population_size, early_stopping, elite_size):
    population = toolbox.population(n=population_size)
    
    # Evaluate the initial population
    fitnesses = list(toolbox.map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    best_fitness = -float("inf")
    stagnation_counter = 0

    # Define Statistics & Logging
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("max", np.max)
    stats.register("mean", np.mean)

    logbook = tools.Logbook()
    logbook.header = ["gen", "max", "mean"]

    for gen in range(n_generations):
        # Select offspring and apply genetic operators
        offspring = toolbox.select(population, len(population) - elite_size)
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.7:  # 70% crossover probability
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < 0.2:  # 20% mutation probability
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Recalculate fitness for modified individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Apply elitism: Retain best individuals
        elite = tools.selBest(population, elite_size)
        population[:] = elite + offspring

        # Track statistics
        record = stats.compile(population)
        logbook.record(gen=gen, **record)
        print(logbook.stream)

        # Early stopping
        current_best = record["max"]
        if current_best <= best_fitness:
            stagnation_counter += 1
        else:
            best_fitness = current_best
            stagnation_counter = 0

        if stagnation_counter >= early_stopping:
            print("Early stopping triggered.")
            break

    # Return best feature set
    best_individual = tools.selBest(population, 1)[0]
    selected_features = np.where(np.array(best_individual) == 1)[0]
    
    return selected_features

# Run the Genetic Algorithm
best_features = run_ga(n_generations=250, population_size=600, early_stopping=50, elite_size=30)
print("BEST FEATURES:", X_train.columns[best_features].tolist())
