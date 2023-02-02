# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 13:48:23 2023

Takes the path to the folder which stores screenshots and outputs the prediction
for current round winner and probability

@author: tchowdh
"""

import os
import time

def read_directory(path, time_delay = 1):
    """
    Monitors the directory given for new files. If a new file is detected then
    the path to this file is output and the function stops.

    Parameters
    ----------
    path : str
        The path to where the leaderboard screenshots will be saved.

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
        
def obtain_prediction():
    """
    Runs directory reader, feeds any new image into ML model and outputs the
    predicted round winner and confidence

    Returns
    -------
    winning_

    """