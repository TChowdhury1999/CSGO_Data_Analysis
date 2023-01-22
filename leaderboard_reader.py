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

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# start by reading score info


def get_score(hsv_img):
    """Function that extracts the score from screengrab image of leader board
    output is a list with rounds won
    """

    rounds = []
    start_x = 804
    pixel_y = 556
    separation_x = 20
    first_half = 8
    half_separation = 22
    minimum_saturation = 30
    minimum_value = 150
    ct_minimum_hue, ct_maximum_hue = [99, 109]
    t_minimum_hue, t_maximum_hue = [17, 28]

    for round_ in range(16):
        if round_ < first_half:
            pixel_x = start_x + round_ * separation_x
        else:
            pixel_x = start_x + half_separation + round_ * separation_x

        hue, saturation, value = hsv_img[pixel_y, pixel_x]

        if (saturation > minimum_saturation) and (value > minimum_value):

            if ct_minimum_hue <= hue <= ct_maximum_hue:
                rounds.append("ct")
            elif t_minimum_hue <= hue <= t_maximum_hue:
                rounds.append("t")

    return rounds


def show_img(img):
    cv2.imshow("ImageWindow", img)
    cv2.waitKey(0)


def get_time(hsv_img):
    """
    Retrieves time from image - if no time found then returns a tuple
    (None, Est_time) where Est_time is a guessed time based on number of kills
    in the round so far
    """

    # convert image to HSV
    lower = np.array([13, 20, 36])
    upper = np.array([17, 117, 192])
    img = cv2.inRange(hsv_img, lower, upper)
    cropped_img = img[279:294, 1382:1421]
    larger_img = cv2.resize(cropped_img, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    invert_img = cv2.bitwise_not(larger_img)
    out_below = pytesseract.image_to_string(
        invert_img, lang="eng", config="--psm 7 -c tessedit_char_whitelist=0123456789:"
    )

    return out_below


def get_k_d_a(img):
    """
    Get kills, deaths and assists from img at img path

    """

    # initialise function by reading in comparison array used to read digits
    comparison_array = np.load("images/digit_images/digit_arrays/comparison_array.npy")
    # create inverse comparison array where there is 0 instead of -1
    inverse_comparison_array = np.where(comparison_array == -1, 0, comparison_array)

    # initialise dataframes for both teams to store the info
    player_df = pd.DataFrame([[0, 0, 0] for i in range(1, 11)], columns=["Kills", "Assists", "Deaths"])

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
                cell_x = (1 - team) * t_start_x + team * ct_start_x + stat * sep_x
                cell_y = (1 - team) * t_start_y + team * ct_start_y + player * sep_y

                # reduce image to this cell
                cell_img = img[cell_y : cell_y + len_y, cell_x : cell_x + len_x]

                # perform otsu thresholding to only extract a solid background with
                # digit on top
                # convert to greyscale
                cell_img = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
                # apply binary otsu threshold
                _, digit_array = cv2.threshold(cell_img, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # create inverse digit array which has -1 instead of 0
                inverse_digit_array = np.where(digit_array == 0, -1, digit_array)

                # now find out what number this is using the comparison array
                # logic here is that the array is flatten into a 1D array
                # then it is tiled (repeated) until it is same size as comparison
                # array
                # we then multiply these two arrays

                flat_array = digit_array.flatten()
                tile_array = np.tile(flat_array, (51, 1))
                weight_array = comparison_array * tile_array
                weights = weight_array.sum(axis=1)

                # repeat this process with inverses

                inverse_flat_array = inverse_digit_array.flatten()
                inverse_tile_array = np.tile(inverse_flat_array, (51, 1))
                inverse_weight_array = inverse_comparison_array * inverse_tile_array
                inverse_weights = inverse_weight_array.sum(axis=1)

                # sum these weight, then the index of the max weight is the
                # number we are looking for
                number = (weights + inverse_weights).argmax()

                # add this number to player dataframe
                player_df.iloc[player + (team * 5), stat] = number

    return player_df


def get_player_alive(img):
    """
    Returns a list that indicates which players are dead or alive
    """

    # initialise by reading in skull array used to compare to check if player
    # is dead
    skull_array = np.load("images/image_templates/skull_array.npy")
    # create inverse skull_array also
    inverse_skull_array = np.where(skull_array == -1, 0, skull_array)

    # list that will contain alive (1) or dead (0) for each player
    alive_list = []

    # size and position of arrays
    start_x = 592
    len_x = 20

    ct_start_y = 377
    t_start_y = 611
    sep_y = 26
    len_y = 22

    for team in range(2):
        for player in range(5):

            # calculate coordinates
            cell_y = ct_start_y * (1 - team) + t_start_y * team + player * sep_y

            # reduce image to this cell
            cell_img = img[cell_y : cell_y + len_y, start_x : start_x + len_x]

            # perform otsu thresholding to only extract a solid background with
            # skull image on top (or no image)
            # convert to greyscale
            cell_img = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
            # apply binary otsu threshold
            _, cell_array = cv2.threshold(cell_img, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # create inverse cell array which has -1 instead of 0
            inverse_cell_array = np.where(cell_array == 0, -1, cell_array)

            weight = (cell_array * skull_array).sum() + (inverse_cell_array * inverse_skull_array).sum()

            # weight above 300 indicates dead
            alive_list.append(weight < 300)

    return alive_list


def create_input(img_path):
    """
    Provide the path to an image of the leaderboard and this function outputs
    the as a dataframe the input for the ML model.
    It has the same columns as the feature_df used to train the ML model
    """

    # read in the image
    img = cv2.imread(img_path)

    # some functions require hsv image. To avoid multiple conversions, lets
    # just convert to hsv once here
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    pass
