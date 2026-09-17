# -*- coding: utf-8 -*-
"""
Created on 2026-09-17
@author: Ruowen Xiao
"""

# %% import packages
import pandas as pd
import requests

# %% Fetch data
df = pd.read_csv("https://ourworldindata.org/explorers/food-footprints.csv?v=1&csvType=full&useColumnShortNames=true&hideControls=true&Commodity+or+Specific+Food+Product=Commodity&Environmental+Impact=Carbon+footprint&Kilogram+%2F+Protein+%2F+Calories=Per+kilogram&By+stage+of+supply+chain=false", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})
metadata = requests.get("https://ourworldindata.org/explorers/food-footprints.metadata.json?v=1&csvType=full&useColumnShortNames=true&hideControls=true&Commodity+or+Specific+Food+Product=Commodity&Environmental+Impact=Carbon+footprint&Kilogram+%2F+Protein+%2F+Calories=Per+kilogram&By+stage+of+supply+chain=false").json()

# %% 
