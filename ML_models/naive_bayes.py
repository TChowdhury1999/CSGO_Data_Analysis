# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 22:46:27 2023

Author: Tanjim Chowdhury

Train & test Naive Bayes ML Model
"""


import pandas as pd
import git
from ML_functions import buildROC
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics



# Load the dataset

repo = git.Repo('.', search_parent_directories=True)
data = pd.read_pickle(repo.working_tree_dir + "/features_dfs/reduced_df.pkl")

# Split the data set into training and test splits (90/10 split used)

x_train, x_test, y_train, y_test = train_test_split(data.drop('Target', axis=1), data.Target, test_size=0.10, random_state=0)

# Make an instance of Naive Bayes model
Naive_Bayes = GaussianNB()

# fit the model to our data
Naive_Bayes.fit(x_train, y_train)

# obtain predictions on test set
predictions = Naive_Bayes.predict(x_test)
y_prob = Naive_Bayes.predict_proba(x_test)

# score the model
score = Naive_Bayes.score(x_test, y_test)
print(score)
cm = metrics.confusion_matrix(y_test, predictions)
print(cm)

# display ROC curve
buildROC(y_test, y_pred = y_prob[:,1])


# for reference
# Score of 0.75
# AUC of 0.82