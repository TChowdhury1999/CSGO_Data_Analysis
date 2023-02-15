# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 13:48:23 2023

Takes the path to the folder which stores screenshots and outputs the prediction
for current round winner and probability

@author: tchowdh
"""

import os
import time
import pickle
import git
import leaderboard_reader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# load in ML models
repo = git.Repo(".", search_parent_directories=True)
scaler = pickle.load(open(repo.working_tree_dir + "/ML_models/scaler.sav", "rb"))
PCA = pickle.load(open(repo.working_tree_dir + "/ML_models/PCA.sav", "rb"))
xgbTree = pickle.load(open(repo.working_tree_dir + "/ML_models/xgbTree.sav", "rb"))


def read_directory(path, time_delay = 1):
    """
    Monitors the directory given for new files. If a new file is detected then
    the path to this file is output and the function stops.

    Parameters
    ----------
    path : str
        The path to where the leaderboard screenshots will be saved.
        
    time_delay : int
        Time between each directory check

    Returns
    -------
    img_path : str
        The path to the latest image in the directory.

    """
    
    # create a set to store the paths to initial files
    initial_file_set = set(os.listdir(path))
    
    while True:
        # create a set of current files
        current_file_set = set(os.listdir(path))
        
        # check for new files using the difference of sets
        new_file_set = current_file_set - initial_file_set
        
        if len(new_file_set) != 0:
            # there is a new file so return the path to it
            return( path + '/' + new_file_set.pop() )
        
        # if there is no new file, then wait 3 seconds before scanning again
        time.sleep(time_delay)
        
def obtain_prediction(image_directory_path, PCA=PCA, scaler=scaler, xgbTree=xgbTree, time_delay=1):
    """
    Runs directory reader, feeds any new image into ML model and outputs the
    predicted round winner and confidence
    
    Parameters
    ----------
    PCA : sklearn PCA model
    
    scaler : sklearn Standard Scaler model
    
    xgbTree : xgBoost decision tree model
    
    image_directory_path : str
        The path to where the leaderboard screenshots will be saved.
        
    time_delay : int
        Time between each directory check

    Returns
    -------
    predicted_round_outcome : tuple
        Tuple with first element being winning team and second element is the
        confidence expressed as a percentage
        
    current_round : int
        The current round so that user knows the correct round has been measured

    """
    
    # get new leaderboard image path
    img_path = read_directory(image_directory_path, time_delay)
    
    # pass this path to the reader which outputs a df for the ML models
    input_df = leaderboard_reader.create_input(img_path)
    current_round = int(input_df.Round)
    
    # pass this df to the scaler and pca models for dimensionality reduction
    scaled_df = scaler.transform(input_df)
    reduced_df = PCA.transform(scaled_df)
    
    # pass this onto boosted decision tree to get output
    # this returns [P(winning team = CT), P(winning team = T)]
    tree_outcome = xgbTree.predict_proba(reduced_df)
    
    
    return tree_outcome, current_round