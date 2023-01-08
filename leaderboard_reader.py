# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 17:57:31 2023

Author: Tanjim

Script that takes a screen grab of the CSGO leaderboard and extracts the info
that is needed to feed into the ML model.
"""


import cv2
import numpy as np
import pytesseract

# start by reading score info

def get_score(img_path):
    """ Function that extracts the score from screengrab image of leader board
        img_path is a str path to that img
    """
    pass

def show_img(img):
    cv2.imshow('ImageWindow', img)
    cv2.waitKey(0)

def img_preprocess_for_text(img_path):
    """ Uses opencv functions to preprocess images for OCR """
    
    img = cv2.imread(img_path)
    
    # convert to greyscale
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    
    # gaussian blur
    # blur_img = cv2.GaussianBlur(grey_img, (5,5), 0)
    
    # threshold
    
    #binary_img = cv2.adaptiveThreshold(grey_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,3,2)

    gray, img_bin = cv2.threshold(grey_img,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    gray = cv2.bitwise_not(img_bin)

    return gray


img_path = "leaderboard_sample_images/test1.png"
processed_img = img_preprocess_for_text(img_path)
cropped_img = processed_img[522:550, 508:573]
invert_img = cv2.bitwise_not(cropped_img)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
out_below = pytesseract.image_to_string(invert_img, config='--psm 7 -c tessedit_char_whitelist=0123456789')
