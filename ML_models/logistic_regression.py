# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 15:57:57 2023

Author: Tanjim Chowdhury

Train & test logistic regression ML model

"""

import pandas as pd
import git
import pickle
from ML_functions import buildROC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV


# Load the dataset

repo = git.Repo(".", search_parent_directories=True)
data = pd.read_pickle(repo.working_tree_dir + "/features_dfs/reduced_df.pkl")

# Split the data set into training and test splits (90/10 split used)
x_train, x_test, y_train, y_test = train_test_split(
    data.drop("Target", axis=1), data.Target, test_size=0.10, random_state=0
)


# Make an instance of Logistic Regression model and
# define parameters
logisticRegr = LogisticRegression()
solvers = ["newton-cg", "lbfgs", "liblinear", "sag"]
penalty = ["l2"]
c_values = [100, 10, 1.0, 0.1, 0.01]


# define grid search
grid = dict(solver=solvers, penalty=penalty, C=c_values)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(
    estimator=logisticRegr, param_grid=grid, n_jobs=-1, cv=cv, scoring="accuracy", error_score=0
)
grid_result = grid_search.fit(x_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_["mean_test_score"]
stds = grid_result.cv_results_["std_test_score"]
params = grid_result.cv_results_["params"]
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# best results were
# Best: 0.773150 using {'C': 0.01, 'penalty': 'l2', 'solver': 'liblinear'}

# now train model with best params
logisticRegr = LogisticRegression(**grid_result.best_params_)
logisticRegr.fit(x_test, y_test)

predictions = logisticRegr.predict(x_test)
y_prob = logisticRegr.predict_proba(x_test)

score = logisticRegr.score(x_test, y_test)
print(score)
cm = metrics.confusion_matrix(y_test, predictions)
print(cm)


# display ROC curve
buildROC(y_test, y_pred=y_prob[:, 1])

# for reference
# Score of 0.78
# AUC of 0.86

# save the model
filename = "logisticRegr.sav"
pickle.dump(logisticRegr, open(filename, "wb"))
