# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 13:52:18 2022

@author: Tanjim
"""

# -*- coding: utf-8 -*-
"""
Created on Mon 26 Dec 2022
Author: Tanjim Chowdhury

Loads in the match DataFrames, combines them and format them so they can train
a ML Model.

Then create additional features that the ML could use

"""


import pandas as pd
import numpy as np
import os

""" DataFrame Loading """

def read_pkl(path):
    """
    Function that reads in the pickle dataframe at path
    This also adds the match ID as a column 
    """
    loaded_df = pd.read_pickle(path)
    match_ID = path[10:-7]
    loaded_df["match_ID"] = match_ID
    return loaded_df


# load in all the dataframes into one dataframe

match_df_paths = ["match_dfs/"+f for f in os.listdir("match_dfs/") if f.endswith(".pickle")]
combined_df = pd.concat(map(read_pkl, match_df_paths))

""" Feature Engineering """

# engineer new features that could be calculated by at a random timestamp
# information available is each row but NOT previous rows
# except is how rounds won/score has evolved over rounds

# first add min/avg/max of (k/d/a) per round/winning round of total/alive players for each team

# kills



# add a pistol round marker

# add a consecutive round win/loss since pistol counter