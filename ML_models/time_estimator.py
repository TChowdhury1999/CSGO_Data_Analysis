# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 12:57:17 2023

@author: Tanjim

This scripts creates a quick linear regression model to predict time based on 
the other features as time is only known to the player sometimes (before bomb
                                                                  plant)
"""


import pickle
import pandas as pd
import git
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# load in the features dataframe
repo = git.Repo(".", search_parent_directories=True)
data = pd.read_pickle(repo.working_tree_dir + "/features_dfs/features_df.pkl")

# remove some columns
# data = data.drop(["match_ID", "team1_won_round", "team2_won_round"], axis=1)
corrmat = data.corr()
target = data.Time
data = data[["team1_players_alive", "team2_players_alive"]]


# Split the data set into training and test splits (90/10 split used)
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.10, random_state=0)

# Make linear regression model instance
linear_regression = LinearRegression()

# fit to the data
linear_regression.fit(x_train, y_train)

# score model
score = linear_regression.score(x_test, y_test)

# save the model
filename = "time_estimator.sav"
pickle.dump(linear_regression, open(filename, "wb"))
