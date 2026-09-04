# -*- coding: utf-8 -*-
"""
Created on 2026-09-03
@author: Ruowen Xiao
"""

# %% Import Packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.optimize import minimize

# %% Read Data
data_path = "./data/bird_count.csv"
df = pd.read_csv(data_path)
df = df.sort_values(by="yr")

# %% Extract Data
years = df["yr"].to_numpy(dtype=int)
years_centered = years - np.min(years)
counts = df["count"].to_numpy(dtype=int)

# %% Model: Poisson GLM
def poisson_glm(x, y):
    def log_likelihood(coef):
        beta0, beta1 = coef
        eta = beta0 + beta1 * x
        return -np.sum(-np.exp(eta) + y * eta)
    result = minimize(log_likelihood, x0=[0, 0])
    # print(result)
    return result.x

# %% Calculate Coefficients
beta0, beta1 = poisson_glm(x=years_centered, y=counts)
print(beta0, beta1)

# %% Simulation

# %% Plot
fig, ax = plt.subplots()
ax.scatter(years, counts)

# %% Write Data
