# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 12:19:02 2022

Author: Tanjim Chowdhury

Loads in the dataframe with all the engineered features and performs 
PCA to reduce dimensions

"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# load in the features dataframe
features_df = pd.read_pickle("features_dfs/features_df.pkl")


# function below to explore validity of using PCA


def plot_3d_points(X_scaled):
    # returns a 3D scatter plot of 3 PCA columns when only using 3 components

    pca_3 = PCA(n_components=3, random_state=1)
    pca_3.fit(X_scaled)
    X_pca_3 = pca_3.transform(X_scaled)

    # plot
    ax = plt.axes(projection="3d")
    ax.scatter(X_pca_3[:, 0], X_pca_3[:, 1], X_pca_3[:, 2], s=10, c=Y, alpha=0.6)
    plt.show()


# PCA

# prepare dataframe by dropping match_id and only include one target column
Y = features_df.team1_won_round
X = features_df.drop(["match_ID", "team2_won_round", "team1_won_round"], axis=1)

# scale the features so we can conduct PCA
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

# reduce to features that explain 95% of variance
pca_all = PCA(n_components=0.95, random_state=1)
pca_all.fit(X=X_scaled)
X_pca_95 = pca_all.transform(X_scaled)

# plt.plot(np.cumsum(pca_all.explained_variance_ratio_*100))

# create new dataframe with PCA components and target column
# add target column
reduced_arr = np.c_[X_pca_95, Y]
reduced_df = pd.DataFrame(reduced_arr, columns=[f"PCA_{i}" for i in range(1, X_pca_95.shape[1] + 1)] + ["Target"])

# save dataframe, scaler and PCA model
reduced_df.to_pickle("features_dfs/reduced_df.pkl")
pickle.dump(scaler, open("ML_models/scaler.sav", "wb"))
pickle.dump(pca_all, open("ML_models/PCA.sav", "wb"))
