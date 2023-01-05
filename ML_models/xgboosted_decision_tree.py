# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 20:19:30 2023

Author: Tanjim Chowdhury

Train and test gradient boosted decision tree
"""


import pandas as pd
import git
import pickle
import xgboost as xgb
from ML_functions import buildROC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV


# Load the dataset

repo = git.Repo('.', search_parent_directories=True)
data = pd.read_pickle(repo.working_tree_dir + "/features_dfs/reduced_df.pkl")

# Split the data set into training and test splits (90/10 split used)
x_train, x_test, y_train, y_test = train_test_split(data.drop('Target', axis=1), data.Target, test_size=0.10, random_state=0)

# Make an instance of KNN model and
# define parameters
xgbTree = xgb.XGBClassifier(tree_method="hist")
n_estimators = [1000]
learning_rate = [0.1]
subsample = [1.0]
max_depth = [9]


# define grid search
grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
cv = RepeatedStratifiedKFold(n_splits=1, n_repeats=1, random_state=1)
grid_search = GridSearchCV(estimator=xgbTree, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0,verbose=10)
grid_result = grid_search.fit(x_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
# best results were
# Best: 0.878431 using {'learning_rate': 0.1, 'max_depth': 9, 'n_estimators': 1000, 'subsample': 1.0}

# now train model with best params
xgbTree = xgb.XGBClassifier(**grid_result.best_params_)
xgbTree.fit(x_train, y_train)

predictions = xgbTree.predict(x_test)
y_prob = xgbTree.predict_proba(x_test)

score = xgbTree.score(x_test, y_test)
print(score)
cm = metrics.confusion_matrix(y_test, predictions)
print(cm)


# display ROC curve
buildROC(y_test, y_pred = y_prob[:,1])

# for reference
# Score of 0.88
# AUC of 0.95

# save the model
filename = 'xgbTree.sav'
pickle.dump(xgbTree, open(filename, 'wb'))
