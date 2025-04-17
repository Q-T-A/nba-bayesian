import pymc as pm
import numpy as np
import arviz as az

def fit_hierarchical_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    elo: np.ndarray,
    coords: dict
) -> (pm.Model, az.InferenceData):
    """
    Build and sample a hierarchical Bayesian model where team effects
    are informed by recent ELO ratings.

    Parameters:
    - X_train: np.ndarray, shape (n_games, n_features)
    - y_train: np.ndarray, shape (n_games,)
    - home_idx: np.ndarray, shape (n_games,), indices of home teams
    - away_idx: np.ndarray, shape (n_games,), indices of away teams
    - elo: np.ndarray, shape (n_teams,), recent ELO of each team
    - coords: dict, mapping dimension names to labels for ArviZ

    Returns:
    - model: pymc.Model
    - trace: arviz.InferenceData
    """
    n_teams = len(elo)
    n_features = X_train.shape[1]

    with pm.Model(coords=coords) as model:
        # Hyperpriors for ELO-informed team effects
        global_mu = pm.Normal("global_mu", mu=100, sigma=15)
        beta_elo = pm.Normal("beta_elo", mu=0, sigma=1)
        sigma_team = pm.HalfNormal("sigma_team", sigma=5)

        # Team-level effects: mu depends on elo
        team_effect = pm.Normal(
            "team_effect",
            mu=global_mu + beta_elo * elo,
            sigma=sigma_team,
            dims="team"
        )

        # Global intercept and feature coefficients
        intercept = pm.Normal("intercept", mu=0, sigma=10)
        betas = pm.Normal(
            "betas", mu=0, sigma=1,
            shape=n_features,
            dims="feature"
        )

        # Game-level expected value
        mu_game = (
            intercept
            + team_effect[home_idx]     # home team contribution
            - team_effect[away_idx]     # away team drag
            + pm.math.dot(X_train, betas)
        )

        # Observation noise
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=10)

        # Likelihood
        y_obs = pm.Normal(
            "y_obs", mu=mu_game, sigma=sigma_obs,
            observed=y_train,
            dims="game"
        )

        # Sampling
        trace = pm.sample(
            draws=1000,
            tune=1000,
            target_accept=0.9,
            return_inferencedata=True
        )

    return model, trace
