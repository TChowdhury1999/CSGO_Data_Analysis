# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 12:12:48 2023

@author: Tanjim

Using PCA scaling coefficients and feature importance from ML model, find the 
importance of the original features in the model

"""


import pickle
import git
import numpy as np
import pandas as pd


# load in pca, ml model and original input_df for column names
repo = git.Repo(".", search_parent_directories=True)

PCA = pickle.load(open(repo.working_tree_dir + "/ML_models/PCA.sav", "rb"))
xgbTree = pickle.load(open(repo.working_tree_dir + "/ML_models/xgbTree.sav", "rb"))

column_names = pickle.load(open(repo.working_tree_dir + "/features_dfs/features_df.pkl", "rb")).columns.drop(
    ["match_ID", "team2_won_round", "team1_won_round"]
)

# obtain importance of each PCA variable
pca_importance = xgbTree.get_booster().get_score(importance_type="gain")
pca_importance = pd.Series(pca_importance, name='pca_importance')

# obtain coefficients for each PCA feature
coefficients = PCA.components_.T
coefficient_matrix = pd.DataFrame(coefficients, columns=[f"PCA_{i}" for i in range(1,coefficients.shape[1]+1)], index=column_names)**2

# propogate importance to the components
importance_matrix = coefficient_matrix * pca_importance

# sum the importance of each original feature
importance = importance_matrix.sum(axis=1)
