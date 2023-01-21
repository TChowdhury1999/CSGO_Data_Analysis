# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 12:57:17 2023

@author: Tanjim

This scripts creates a quick linear regression model to predict time based on 
the other features as time is only known to the player sometimes (before bomb
                                                                  plant)
"""


import numpy as np
import pandas as pd
import git


# load in the features dataframe
repo = git.Repo(".", search_parent_directories=True)
data = pd.read_pickle(repo.working_tree_dir + "/features_dfs/features_df.pkl")