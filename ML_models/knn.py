# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 19:19:03 2023

@author: Tanjim

Train and test k-nearest neighbours machine learning model
"""


import pandas as pd
import git
import pickle
from ML_functions import buildROC
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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
kNearest = KNeighborsClassifier()
n_neighbors = range(1, 21, 2)
weights = ['uniform', 'distance']
metric = ['euclidean', 'manhattan', 'minkowski']


# define grid search
grid = dict(n_neighbors=n_neighbors,weights=weights,metric=metric)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=kNearest, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0,verbose=10)
grid_result = grid_search.fit(x_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
# best results were
# Best: 0.773150 using {'metric': 'manhattan', 'n_neighbors': 1, 'weights': 'uniform'}

# now train model with best params
kNearest = KNeighborsClassifier(**grid_result.best_params_)
kNearest.fit(x_train, y_train)

predictions = kNearest.predict(x_test)
y_prob = kNearest.predict_proba(x_test)

score = kNearest.score(x_test, y_test)
print(score)
cm = metrics.confusion_matrix(y_test, predictions)
print(cm)


# display ROC curve
buildROC(y_test, y_pred = y_prob[:,1])

# for reference
# Score of 0.92
# AUC of 0.92

# save the model
filename = 'knn.sav'
pickle.dump(kNearest, open(filename, 'wb'))
