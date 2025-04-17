import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_parquet("datasets copy/q1_m.parquet")

# Create the target column (sum of final scores)
df["TOTAL_POINTS"] = df["HFINAL"] + df["AFINAL"]

# Drop columns you don't want as predictors
df = df.drop(columns=["HFINAL", "AFINAL", "SPREAD", "TOTAL"], errors='ignore')  # `errors='ignore'` handles if 'TOTAL' doesn't exist

# Automatically select all numeric features except the target
target_name = "TOTAL_POINTS"
all_features = df.select_dtypes(include=[np.number]).columns.drop(target_name)

# Drop missing values and sample
df_clean = df[all_features.tolist() + [target_name]].dropna().sample(n=1000, random_state=42)

# Standardize features
X = df_clean[all_features]
y = df_clean[target_name]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Bayesian linear regression with PyMC
with pm.Model() as model:
    X_data = pm.Data("X", X_scaled)
    y_data = pm.Data("y", y.values)

    intercept = pm.Normal("Intercept", mu=0, sigma=10)
    coefs = pm.Normal("Betas", mu=0, sigma=1, shape=X_scaled.shape[1])
    sigma = pm.HalfNormal("Sigma", sigma=10)

    mu = intercept + pm.math.dot(X_data, coefs)
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_data)

    trace = pm.sample(1000, tune=1000, target_accept=0.9, return_inferencedata=True, cores=1)


# Summarize and plot
az.summary(trace, var_names=["Intercept", "Betas", "Sigma"], round_to=2)
az.plot_trace(trace, var_names=["Intercept", "Betas", "Sigma"])
plt.tight_layout()
plt.show()
# Posterior predictive checks
with model:
    ppc = pm.sample_posterior_predictive(trace, var_names=["y_obs"], random_seed=42)

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Get posterior predictive means
y_pred = ppc.posterior_predictive["y_obs"].mean(dim=["chain", "draw"]).values


# KDE plot of actual vs predicted
plt.figure(figsize=(10, 5))
sns.kdeplot(y, label="Actual", color="black")
sns.kdeplot(y_pred, label="Predicted", color="blue")
plt.xlabel("Total Points")
plt.ylabel("Density")
plt.title("Posterior Predictive Check: Actual vs Predicted TOTAL_POINTS")
plt.legend()
plt.show()

# Compute R²
r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
print(f"Posterior Predictive R²: {r2:.3f}")
