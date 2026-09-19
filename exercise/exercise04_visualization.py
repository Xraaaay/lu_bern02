# -*- coding: utf-8 -*-
"""
Created on 2026-09-17
@author: Ruowen Xiao
"""

# %% import packages
import matplotlib.pyplot as plt
import pandas as pd
import requests

# %% Fetch data
df = pd.read_csv("https://ourworldindata.org/explorers/food-footprints.csv?v=1&csvType=full&useColumnShortNames=true&hideControls=true&Commodity+or+Specific+Food+Product=Commodity&Environmental+Impact=Carbon+footprint&Kilogram+%2F+Protein+%2F+Calories=Per+kilogram&By+stage+of+supply+chain=false", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})
metadata = requests.get("https://ourworldindata.org/explorers/food-footprints.metadata.json?v=1&csvType=full&useColumnShortNames=true&hideControls=true&Commodity+or+Specific+Food+Product=Commodity&Environmental+Impact=Carbon+footprint&Kilogram+%2F+Protein+%2F+Calories=Per+kilogram&By+stage+of+supply+chain=false").json()

# Data Process
title = metadata["chart"]["title"]
emissions = "ghg_emissions_per_kilogram__poore__and__nemecek__2018"

# %% Bad Plot
fig, ax = plt.subplots()
ax.bar(df["entity"], df[emissions])

ax.set_xlabel("Type of food")
ax.tick_params(axis="x", labelrotation=90)
ax.set_ylabel("Greenhouse gas emissions(kg)")
ax.set_title(title)
plt.show()

# %% Good Plot
df_ordered = df.sort_values(by=emissions)

fig, ax = plt.subplots(figsize=(12, 12))

colors = 2 * list(plt.colormaps["tab20c"].colors)  # type: ignore
bars = ax.barh(df_ordered["entity"], df_ordered[emissions], color=colors[:len(df)])
ax.bar_label(bars, fmt="%.2f kg", padding=3)

# Remove all the spines
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(axis="x", bottom=False, labelbottom=False)

ax.set_title(title)
plt.show()

# %%
