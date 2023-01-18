# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 17:57:31 2023

Author: Tanjim

Script that takes a screen grab of the CSGO leaderboard and extracts the info
that is needed to feed into the ML model.
"""


import cv2
import numpy as np
import pandas as pd
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# start by reading score info

def get_score(img_path):
    """ Function that extracts the score from screengrab image of leader board
        img_path is a str path to that img
        
        pseudo code - simply extract the colour at pixel positions near middle
        of score board
        
    """
    pass

def show_img(img):
    cv2.imshow('ImageWindow', img)
    cv2.waitKey(0)

def get_time(img_path):
    """
    Retrieves time from image - if no time found then returns a tuple 
    (None, Est_time) where Est_time is a guessed time based on number of kills
    in the round so far
    """
    
    # read in image
    img = cv2.imread(img_path)
    # convert image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([13,20,36])
    upper = np.array([17,117,192])
    img = cv2.inRange(hsv, lower, upper)
    cropped_img = img[279:294, 1382:1421]
    larger_img = cv2.resize(cropped_img, (0,0), fx=3, fy=3, interpolation = cv2.INTER_NEAREST)
    invert_img = cv2.bitwise_not(larger_img)
    out_below = pytesseract.image_to_string(invert_img, lang = "eng", config='--psm 7 -c tessedit_char_whitelist=0123456789:')
    
    return out_below

def get_k_d_a(img_path):
    """
    Get kills, deaths and assists from img at img path
    
    """

    # initialise function by reading in comparison array used to read digits
    comparison_array = np.load("images/digit_images/digit_arrays/comparison_array.npy")
    # create inverse comparison array where there is 0 instead of -1
    inverse_comparison_array = np.where(comparison_array == -1, 0, comparison_array)
    
    # img_path = "digit_images/3_CT.png"
    
    # read in image
    img = cv2.imread(img_path)
    
    
    # initialise dataframes for both teams to store the info
    player_df = pd.DataFrame([[0, 0, 0] for i in range(1,11)], columns=["Kills", "Assists", "Deaths"])
    
    ct_start_x = 1224
    ct_start_y = 377
    
    t_start_x = 1224
    t_start_y = 611
    
    # separation of cells on grid here
    sep_x = 35
    sep_y = 26
    
    # height and width of cells here
    len_x = 32
    len_y = 21
    
    
    # loop over each cell in the grid (3 stats by 5 players in 2 teams)
    for team in range(2):
        for player in range(5):
            for stat in range(3):
                
                # calculate coordinates of the cell
                cell_x = (1-team)*t_start_x + team*ct_start_x + stat*sep_x
                cell_y = (1-team)*t_start_y + team*ct_start_y + player*sep_y
                
                # reduce image to this cell
                cell_img = img[cell_y:cell_y+len_y, cell_x:cell_x+len_x]
                
                # perform otsu thresholding to only extract a solid background with
                # digit on top
                # convert to greyscale
                cell_img = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
                # apply binary otsu threshold
                _, digit_array = cv2.threshold(cell_img,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                
                # create inverse digit array which has -1 instead of 0
                inverse_digit_array = np.where(digit_array == 0, -1, digit_array)
                
                # now find out what number this is using the comparison array
                # logic here is that the array is flatten into a 1D array
                # then it is tiled (repeated) until it is same size as comparison
                # array
                # we then multiply these two arrays 
    
                flat_array = digit_array.flatten()
                tile_array = np.tile(flat_array, (51,1))
                weight_array = comparison_array * tile_array
                weights = weight_array.sum(axis=1)
                
                # repeat this process with inverses
                
                inverse_flat_array = inverse_digit_array.flatten()
                inverse_tile_array = np.tile(inverse_flat_array, (51,1))
                inverse_weight_array = inverse_comparison_array * inverse_tile_array
                inverse_weights = inverse_weight_array.sum(axis=1)
                
                # sum these weight, then the index of the max weight is the
                # number we are looking for
                number = (weights+inverse_weights).argmax()
                
                # add this number to player dataframe
                player_df.iloc[player+(team*5),stat] = number
            
            
    return player_df

