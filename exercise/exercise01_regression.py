# -*- coding: utf-8 -*-
"""
Created on 2026-09-01
@author: Ruowen Xiao
"""
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize

# %% Local Regression Function
def local_regression(x_global, y_global, k, x_0):
    # choose k points nearest to x0
    index = np.argsort(np.abs(x_global - x_0))[:k]
    x = x_global[index]
    y = y_global[index]

    # weight
    h = np.max(np.abs(x - x_0))
    K = (1 - np.abs((x - x_0) / h)**3)**3

    # find beta_1 and beta_0
    x_w_mean = np.sum(K * x) / np.sum(K)
    y_w_mean = np.sum(K * y) / np.sum(K)
    beta_1 = np.sum(K * (x - x_w_mean) * (y - y_w_mean)) / np.sum(K * (x - x_w_mean)**2)
    beta_0 = y_w_mean - beta_1 * x_w_mean
    print(f"[{beta_0} {beta_1}]")

    # calculate predicted value and standard deviation
    pred = beta_0 + beta_1 * x_0
    se = np.sqrt(np.sum(K * (y - beta_0 - beta_1 * x)**2) / (k - 2))
    return pred, se

# %% Cross Validation
# TODO
def cross_validate():
    return 15

# %% Load Data
data_path = "./data/pollution_cleaneddata.csv"
df = pd.read_csv(data_path, delimiter=',', dtype=float)
df = df.sort_values(by="POOR")

x = df["POOR"].to_numpy()  # observations of predictor
y = df["MORT"].to_numpy()  # observations of response
k = cross_validate()  # number of neighboring points
x_0_list = [10, 18, 25]  # values for prediction

# %% Calculation
results = [local_regression(x, y, k, x_0) for x_0 in x_0_list]
print(results)

# %% Plot
x_0_grid = np.linspace(x.min(), x.max(), 20)
y_pred = np.array([local_regression(x, y, k, x_0)[0] for x_0 in x_0_grid])
print(x_0_grid)
print(y_pred)

fig, ax = plt.subplots()
ax.scatter(x, y)
ax.plot(x_0_grid, y_pred)

# %% 
# Results
# x_0 = 10: pred = 899.79, se = 59.25
# x_0 = 18: pred = 956.16, se = 59.37
# x_0 = 25: pred = 1011.09, se = 62.67
