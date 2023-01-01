# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 15:57:57 2023

Author: Tanjim Chowdhury

Train & test logistic regression ML model

"""

import pandas as pd
import git
from ML_functions import buildROC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics



# Load the dataset

repo = git.Repo('.', search_parent_directories=True)
data = pd.read_pickle(repo.working_tree_dir + "/features_dfs/reduced_df.pkl")

# Split the data set into training and test splits (90/10 split used)

x_train, x_test, y_train, y_test = train_test_split(data.drop('Target', axis=1), data.Target, test_size=0.10, random_state=0)

# Make an instance of Logistic Regression model
logisticRegr = LogisticRegression()

# fit the model to our data
logisticRegr.fit(x_train, y_train)

# obtain predictions on test set
predictions = logisticRegr.predict(x_test)
y_prob = logisticRegr.predict_proba(x_test)

# score the model
score = logisticRegr.score(x_test, y_test)
print(score)
cm = metrics.confusion_matrix(y_test, predictions)
print(cm)

# display ROC curve
buildROC(y_test, y_pred = y_prob[:,1])


# for reference
# Score of 0.77
# AUC of 0.86