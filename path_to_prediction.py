# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 13:48:23 2023

Takes the path to the folder which stores screenshots and outputs the prediction
for current round winner and probability

@author: tchowdh
"""

import os
import pickle
import git
import leaderboard_reader
import glob

# load in ML models
repo = git.Repo(".", search_parent_directories=True)
scaler = pickle.load(open(repo.working_tree_dir + "/ML_models/scaler.sav", "rb"))
PCA = pickle.load(open(repo.working_tree_dir + "/ML_models/PCA.sav", "rb"))
xgbTree = pickle.load(open(repo.working_tree_dir + "/ML_models/xgbTree.sav", "rb"))


def get_latest_file(path):
    """
    Returns the file path of the latest file in the directory passed

    Parameters
    ----------
    path : str
        The path to the directory to be checked
        

    Returns
    -------
    latest_path : str
        The path to the latest file in the directory.

    """
    
    # obtain all files in path
    list_of_files = glob.glob(f'{path}/*')
    
    # latest file has maximum of creation time
    latest_file = max(list_of_files, key=os.path.getctime)
    
    return latest_file
        
def obtain_prediction(img_path, PCA=PCA, scaler=scaler, xgbTree=xgbTree):
    """
    Feeds image at img_path into ML model and outputs the
    predicted round winner and confidence as probability (and round number)
    
    Parameters
    ----------
    PCA : sklearn PCA model
    
    scaler : sklearn Standard Scaler model
    
    xgbTree : xgBoost decision tree model
    
    img_path : str
        The path to image of leaderboard

    Returns
    -------
    tree_outcome : list
        List where first element is probability of CT winning, second is prob
        of T winning and final element is round number
        
    """
    
    # pass this path to the reader which outputs a df for the ML models
    input_df = leaderboard_reader.create_input(img_path)
    current_round = int(input_df.Round)
    
    # pass this df to the scaler and pca models for dimensionality reduction
    scaled_df = scaler.transform(input_df)
    reduced_df = PCA.transform(scaled_df)
    
    # pass this onto boosted decision tree to get output
    # this returns [P(winning team = CT), P(winning team = T)]
    tree_outcome = xgbTree.predict_proba(reduced_df)
    
    
    return tree_outcome.tolist()[0] + [current_round]



def write_prediction(result):
    """
    Writes ML model result to a file
    """
    with open('winner_cache.pickle', 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)


def update_prediction(image_directory_path, previous_file, latest_file, PCA, scaler, xgbTree):
   """
   Updates the ML model result in the text file if a new leaderboard image
   file is found
   """
   
   latest_file = get_latest_file(image_directory_path)
   
   if latest_file != previous_file:
       previous_file = latest_file
       outcome = obtain_prediction(latest_file, PCA, scaler, xgbTree)
       write_prediction(outcome)
       print()
       
   else:
       print("No new file")
       pass
   
def read_prediction():
    with open('winner_cache.pickle', 'rb') as handle:
        prediction = pickle.load(handle)
        
    return prediction