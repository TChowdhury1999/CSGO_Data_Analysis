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
    
    # read in image
    img = cv2.imread(img_path)
    
    # start with CT
    # initialise a dataframe to store the info
    ct_df = pd.DataFrame([[i, 0, 0, 0] for i in range(1,6)], columns=["Player", "Kills", "Assists", "Deaths"])
    
    # top left coordinate of CT k/d/a grid here 
    """
    CHANGE THIS SO CELLS ARE MUCH SMALLER - HEIGHT CAN BE VERY REDUCED
    - GET 44 KILLS AND FIND WIDTH
    """
    ct_start_x = 1224
    ct_start_y = 377
    
    # separation of cells on grid here
    sep_x = 35
    sep_y = 26
    
    # height and width of cells here
    len_x = 32
    len_y = 21
    
    # loop over each cell in the grid (3 stats by 5 players)
    for i in range(3):
        for j in range(5):
            
            # calculate coordinates of the cell
            cell_x = ct_start_x + i*sep_x
            cell_y = ct_start_y + i*sep_y
            
            # reduce image to this cell
            cell_img = img[cell_y:cell_y+len_y, cell_x:cell_x+len_x]
            
            # identify background colour of the cell
            # just use the first pixel as it wont have the number information
            # then remove this color from cell
            # because background color can slightly vary, apply colour 
            # reduction
            
            
    
    
    
#########################
#       Testing Area
# Cant seem to get k/d/a numbers with tesseract
# try see if we can used a handdrawn digits/other ML model
# otherwise try template matching

img_path = "leaderboard_sample_images/test1.png"
# processed_img = img_preprocess_for_text(img_path)
img = cv2.imread(img_path)

# convert image to hsv
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# filter based on hsv
lower = np.array([17,56,151])
upper = np.array([37,128,255])
img = cv2.inRange(hsv, lower, upper)

#grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
#gray, img_bin = cv2.threshold(grey_img,250,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#gray = cv2.bitwise_not(img_bin)
# cropped_img = processed_img[520:550, 500:540] # 1st half team 1
cropped_img = img[610:735, 1225:1325] # 1st half team 1
larger_img = cv2.resize(cropped_img, (0,0), fx=10, fy=10, interpolation = cv2.INTER_NEAREST)
# cropped_img = processed_img[520:550, 540:580] # 2nd half team 1
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
dilate = cv2.dilate(larger_img, kernel, iterations=4)
invert_img = cv2.bitwise_not(dilate)
#coords = cv2.findNonZero(invert_img)
#x, y, w, h = cv2.boundingRect(coords)
#final_img = invert_img[y:y+h, x:x+w]

#invert_img = invert_img[0:250, 0:1000]
invert_img = invert_img[0:250, 666:1000]
#invert_img = invert_img[250:500, 0:1000]


out_below = pytesseract.image_to_string(invert_img, lang = "eng", config='outputbase--psm 10 -c tessedit_char_whitelist=0123456789')