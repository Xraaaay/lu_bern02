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
    h = np.max(x) - np.min(x)
    K = (1 - np.abs((x - x_0) / h)**3)**3

    # find beta_1 and beta_0
    def cal_error(args):
        beta_0, beta_1 = args
        y_hat = beta_0 + beta_1 * x
        return np.sum(K * (y - y_hat)**2)

    result = minimize(cal_error, x0=[0, 0])
    beta_0, beta_1 = result.x
    print(f"beta_0 = {beta_0}, beta_1 = {beta_1}")

    # calculate predicted value and standard deviation
    pred = beta_0 + beta_1 * x_0
    se = 0  # TODO
    return pred, se

# %% Cross Validation
def cross_validate():
    return 20

# %% Loda Data
data_path = "./data/pollution_cleaneddata.csv"
df = pd.read_csv(data_path, delimiter=',', dtype=float)
df = df.sort_values(by="POOR")

x = df["POOR"].to_numpy()  # observations of predictor
y = df["MORT"].to_numpy()  # observations of response
k = cross_validate()  # number of neighboring points
x_0_list = [10, 18, 25]  # values for prediction

# %% Plot

# %% Calculation
preds = []
ses = []
for x_0 in x_0_list:
    pred, se = local_regression(x, y, k, x_0)
    preds.append(pred)
    ses.append(se)
print(preds, ses)

# %%
