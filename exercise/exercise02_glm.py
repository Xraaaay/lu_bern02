# -*- coding: utf-8 -*-
"""
Created on 2026-09-03
@author: Ruowen Xiao
"""

# %% Import Packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %% Load Data
data_path = "./data/bird_count.csv"
df = pd.read_csv(data_path)
df = df.sort_values(by="yr")
years = df["yr"].to_numpy(dtype=int)
counts = df["count"].to_numpy(dtype=int)

# %% Model: Poisson Regression
# TODO
def poisson_regression(x, y):
    pass

# %%
