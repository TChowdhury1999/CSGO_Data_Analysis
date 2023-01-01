# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 22:42:57 2023

@author: Tanjim

Useful functions for ML model scripts

"""

import matplotlib.pyplot as plt
from sklearn import metrics


def buildROC(y_test,y_pred, test=True, title="ROC"):
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure(title)
    plt.title('Receiver Operating Characteristic')
    if test:
        plt.plot(fpr, tpr, 'b', label = f'Testing data AUC = {roc_auc}')
    else:
        plt.plot(fpr, tpr, 'r', label = f'Training data AUC = {roc_auc}')        
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')