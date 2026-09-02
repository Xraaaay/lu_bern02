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
def local_regression(x, y, k, x_0):
    # choose k points nearest to x0
    index = np.argsort(np.abs(x - x_0))[:k]
    x = x[index]
    y = y[index]

    # weight
    h = np.max(np.abs(x - x_0))
    K = (1 - np.abs((x - x_0) / h)**3)**3

    # find beta_1 and beta_0
    def cal_error(args):
        beta_0, beta_1 = args
        y_hat = beta_0 + beta_1 * x
        return np.sum(K * (y - y_hat)**2)

    result = minimize(cal_error, x0=[0, 0])
    beta_0, beta_1 = result.x
    # print(f"beta_0 = {beta_0}, beta_1 = {beta_1}")

    # calculate predicted value and standard deviation
    pred = beta_0 + beta_1 * x_0
    se = np.sqrt(np.sum((y - beta_0 - beta_1 * x)**2) / (k - 2))
    return pred, se

# %% Cross Validation
# TODO
def cross_validate():
    return 15

# %% Loda Data
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
