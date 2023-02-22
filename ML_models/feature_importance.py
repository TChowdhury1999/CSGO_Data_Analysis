# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 12:12:48 2023

@author: Tanjim

Using PCA scaling coefficients and feature importance from ML model, find the 
importance of the original features in the model

"""


import pickle
import git



# load in pca and ml model
repo = git.Repo(".", search_parent_directories=True)

PCA = pickle.load(open(repo.working_tree_dir + "/ML_models/PCA.sav", "rb"))
xgbTree = pickle.load(open(repo.working_tree_dir + "/ML_models/xgbTree.sav", "rb"))


