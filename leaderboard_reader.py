# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 17:57:31 2023

Author: Tanjim

Script that takes a screen grab of the CSGO leaderboard and extracts the info
that is needed to feed into the ML model.
"""


import cv2
import pickle
import numpy as np
import pandas as pd

# import pytesseract
import git
from feature_engineering import consecutive_binary_counter

# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


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
    minimum_saturation = 17
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


def create_comp_array():
    """
    Creates comparison array for digit identification
    """

    # path to images
    repo = git.Repo(".", search_parent_directories=True)
    img_path = repo.working_tree_dir + "/images/time_images/"

    # initialise the array
    comparison_array = []

    # coordinates to get boxes

    start_x = 1411
    start_y = 280

    len_x = 8
    len_y = 14

    # load in each image
    for img_number in range(11):

        img = cv2.imread(img_path + f"{img_number}.jpg")

        # crop image to number
        cell_img = img[start_y : start_y + len_y, start_x : start_x + len_x]

        # convert to greyscale
        cell_img = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)

        # apply otsu binarization
        _, time_array = cv2.threshold(cell_img, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # flatten array and add to comparison array
        flat_array = time_array.flatten()
        comparison_array.append(flat_array)

    np.save("images/time_images/time_arrays/time_comparison_array.npy", np.array(comparison_array))


def get_time(img):
    """
    Retrieves time from image
    """
    # time stored here
    time = []
    # there are three digits to read in the time
    # define the starting coordinates in a list here

    start_xs = [1384, 1402, 1411]
    start_y = 280

    len_x = 8
    len_y = 14

    # load in comparison array
    comparison_array = np.load("images/time_images/time_arrays/time_comparison_array.npy")
    # create inverse
    inverse_comparison_array = np.where(comparison_array == -1, 0, comparison_array)

    for digit in range(3):

        # crop image to digit region
        digit_img = img[start_y : start_y + len_y, start_xs[digit] : start_xs[digit] + len_x]
        # convert to grayscale
        grey_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
        # apply otsu binarisation to remove background
        _, digit_array = cv2.threshold(grey_img, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # create inverse binary image where 0 -> -1
        inverse_digit_array = np.where(digit_array == 0, -1, digit_array)

        # create initial weights array
        flat_array = digit_array.flatten()
        tile_array = np.tile(flat_array, (11, 1))
        weight_array = comparison_array * tile_array
        weights = weight_array.sum(axis=1)

        # create the inverse weights array
        inverse_flat_array = inverse_digit_array.flatten()
        inverse_tile_array = np.tile(inverse_flat_array, (11, 1))
        inverse_weight_array = inverse_comparison_array * inverse_tile_array
        inverse_weights = inverse_weight_array.sum(axis=1)

        # sum these weight, then the index of the max weight is the
        # number we are looking for
        total_weights = weights + inverse_weights
        if digit == 0:
            # the first digit can only be 0 or 1 so let's restrict to this
            # (and none)
            digit = np.append(total_weights[:2], total_weights[-1]).argmax()
        else:
            digit = total_weights.argmax()

        if digit == 10:
            return None
        else:
            time.append(digit)

    return time[0] * 60 + time[1] * 10 + time[2]


def get_k_d_a(img):
    """
    Get kills, deaths and assists from img at img path

    """

    # initialise function by reading in comparison array used to read digits
    comparison_array = np.load("images/digit_images/digit_arrays/comparison_array.npy")
    # create inverse comparison array where there is 0 instead of -1
    inverse_comparison_array = np.where(comparison_array == -1, 0, comparison_array)

    # initialise dataframes for both teams to store the info
    player_df = pd.DataFrame([[0, 0, 0] for i in range(1, 11)], columns=["kills", "assists", "deaths"])

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

    return np.array(alive_list).astype(int)


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

    # initialise an empty dataframe here
    # this has the same info as the dataframe we would read during data
    # collection
    # using a dataframe just so we can reuse the code from feature_engineering
    # to recreate the features from this data

    read_df = pd.DataFrame(
        np.zeros((1, 44)),
        columns=["Round", "Time", "team1_score", "team2_score"]
        + [f"player{i}_alive" for i in range(1, 11)]
        + [f"player{i}_kills" for i in range(1, 11)]
        + [f"player{i}_assists" for i in range(1, 11)]
        + [f"player{i}_deaths" for i in range(1, 11)],
    )

    # start with round & scores
    rounds = get_score(hsv_img)

    # team1 is t side
    read_df.loc[0, "team1_score"] = rounds.count("t")
    read_df.loc[0, "team2_score"] = rounds.count("ct")
    read_df.loc[0, "Round"] = len(rounds) + 1

    # get k_d_a
    k_d_a = get_k_d_a(img)

    for player in range(10):
        for stat in ["kills", "assists", "deaths"]:
            read_df.loc[0, f"player{player+1}_{stat}"] = k_d_a[f"{stat}"][player]

    # get player alive
    alive_list = get_player_alive(img)

    for player in range(10):
        read_df.loc[0, f"player{player+1}_alive"] = alive_list[player]

    # now begin creating input dataframe for ML model

    time = get_time(img)
    read_df["Time"] = time

    # create input df that will have the features ready for the PCA model
    input_df = read_df[["Round", "Time", "team1_score", "team2_score"]].copy()
    working_df = pd.DataFrame()

    for player_number in range(1, 11):

        # rounds
        working_df[f"player{player_number}_k_per_round"] = read_df[f"player{player_number}_kills"] / read_df["Round"]
        working_df[f"player{player_number}_a_per_round"] = read_df[f"player{player_number}_assists"] / read_df["Round"]
        working_df[f"player{player_number}_d_per_round"] = read_df[f"player{player_number}_deaths"] / read_df["Round"]

        # winning rounds (score)
        if player_number <= 5:
            score_ = read_df["team1_score"]
        else:
            score_ = read_df["team2_score"]

        # if no rounds won, avoid a zero division by setting won rounds to 1
        score_[score_ == 0] = 1

        working_df[f"player{player_number}_k_per_w_round"] = read_df[f"player{player_number}_kills"] / score_
        working_df[f"player{player_number}_a_per_w_round"] = read_df[f"player{player_number}_assists"] / score_
        working_df[f"player{player_number}_d_per_w_round"] = read_df[f"player{player_number}_deaths"] / score_

    for metric in ["k", "d", "a"]:
        for round_ind in ["", "w_"]:
            # all players for each team
            # team1
            input_df[f"team1_min_{metric}_per_{round_ind}r_of_total_p"] = working_df[
                [f"player{player_number}_{metric}_per_{round_ind}round" for player_number in range(1, 6)]
            ].min(axis=1)
            input_df[f"team1_mean_{metric}_per_{round_ind}r_of_total_p"] = working_df[
                [f"player{player_number}_{metric}_per_{round_ind}round" for player_number in range(1, 6)]
            ].mean(axis=1)
            input_df[f"team1_max_{metric}_per_{round_ind}r_of_total_p"] = working_df[
                [f"player{player_number}_{metric}_per_{round_ind}round" for player_number in range(1, 6)]
            ].max(axis=1)
            # team2
            input_df[f"team2_min_{metric}_per_{round_ind}r_of_total_p"] = working_df[
                [f"player{player_number}_{metric}_per_{round_ind}round" for player_number in range(6, 11)]
            ].min(axis=1)
            input_df[f"team2_mean_{metric}_per_{round_ind}r_of_total_p"] = working_df[
                [f"player{player_number}_{metric}_per_{round_ind}round" for player_number in range(6, 11)]
            ].mean(axis=1)
            input_df[f"team2_max_{metric}_per_{round_ind}r_of_total_p"] = working_df[
                [f"player{player_number}_{metric}_per_{round_ind}round" for player_number in range(6, 11)]
            ].max(axis=1)

            # alive players for each team
            # logic for alive players is to just multiply
            # team 1
            input_df[f"team1_min_{metric}_per_{round_ind}r_of_alive_p"] = (
                working_df[[f"player{player_number}_{metric}_per_{round_ind}round" for player_number in range(1, 6)]]
                .multiply(np.array(read_df[[f"player{player_number}_alive" for player_number in range(1, 6)]]))
                .min(axis=1)
            )
            input_df[f"team1_mean_{metric}_per_{round_ind}r_of_alive_p"] = (
                working_df[[f"player{player_number}_{metric}_per_{round_ind}round" for player_number in range(1, 6)]]
                .multiply(np.array(read_df[[f"player{player_number}_alive" for player_number in range(1, 6)]]))
                .mean(axis=1)
            )
            input_df[f"team1_max_{metric}_per_{round_ind}r_of_alive_p"] = (
                working_df[[f"player{player_number}_{metric}_per_{round_ind}round" for player_number in range(1, 6)]]
                .multiply(np.array(read_df[[f"player{player_number}_alive" for player_number in range(1, 6)]]))
                .max(axis=1)
            )

            # team 2
            input_df[f"team2_min_{metric}_per_{round_ind}r_of_alive_p"] = (
                working_df[[f"player{player_number}_{metric}_per_{round_ind}round" for player_number in range(6, 11)]]
                .multiply(np.array(read_df[[f"player{player_number}_alive" for player_number in range(6, 11)]]))
                .min(axis=1)
            )
            input_df[f"team2_mean_{metric}_per_{round_ind}r_of_alive_p"] = (
                working_df[[f"player{player_number}_{metric}_per_{round_ind}round" for player_number in range(6, 11)]]
                .multiply(np.array(read_df[[f"player{player_number}_alive" for player_number in range(6, 11)]]))
                .mean(axis=1)
            )
            input_df[f"team2_max_{metric}_per_{round_ind}r_of_alive_p"] = (
                working_df[[f"player{player_number}_{metric}_per_{round_ind}round" for player_number in range(6, 11)]]
                .multiply(np.array(read_df[[f"player{player_number}_alive" for player_number in range(6, 11)]]))
                .max(axis=1)
            )

    input_df["team1_players_alive"] = (
        read_df.player1_alive
        + read_df.player2_alive
        + read_df.player3_alive
        + read_df.player4_alive
        + read_df.player5_alive
    )
    input_df["team2_players_alive"] = (
        read_df.player6_alive
        + read_df.player7_alive
        + read_df.player8_alive
        + read_df.player9_alive
        + read_df.player10_alive
    )
    
    # add total/team k/d/a 
    for metric in ['kills', 'deaths', 'assists']:
        
        # team1, team2, total
        working_df[f"team1_{metric[0]}"] = read_df[[f"player{i}_{metric}" for i in range(1,6)]].sum(axis=1)
        working_df[f"team2_{metric[0]}"] = read_df[[f"player{i}_{metric}" for i in range(6,11)]].sum(axis=1)
        working_df[f"total_{metric[0]}"] = read_df[[f"player{i}_{metric}" for i in range(1,11)]].sum(axis=1)
        
        # set equal to 1 where it is 0
        working_df[ working_df[f"team1_{metric[0]}"] ==0 ] =1
        working_df[ working_df[f"team2_{metric[0]}"] ==0 ] =1
        working_df[ working_df[f"total_{metric[0]}"] ==0 ] =1
        
        # alive players only
        # team1, team2, total
        working_df[f"team1_alive_{metric[0]}"] = read_df[[f"player{i}_{metric}" for i in range(1,6)]].multiply(np.array(read_df[[f"player{player_number}_alive" for player_number in range(1, 6)]])).sum(axis=1)
        working_df[f"team2_alive_{metric[0]}"] = read_df[[f"player{i}_{metric}" for i in range(6,11)]].multiply(np.array(read_df[[f"player{player_number}_alive" for player_number in range(6, 11)]])).sum(axis=1)
        
    # add percentage of total/team k/d/a of alive players
    

    for metric in ['k', 'd', 'a']:
            
        # team1
        input_df[f"team1_perc_{metric}_of_team"] = input_df[f"team1_alive_{metric}"]/input_df[f"team1_{metric}"]
        input_df[f"team1_perc_{metric}_of_total"] = input_df[f"team1_alive_{metric}"]/input_df[f"total_{metric}"]

        # team2
        input_df[f"team2_perc_{metric}_of_team"] = input_df[f"team2_alive_{metric}"]/input_df[f"team2_{metric}"]
        input_df[f"team2_perc_{metric}_of_total"] = input_df[f"team2_alive_{metric}"]/input_df[f"total_{metric}"]
    

    # add time to the input dataframe
    # depending on if a time is actually visible, it may have to be inferred
    if time == None:
        # load linear regression time model
        model_filename = "ML_models/time_estimator.sav"
        time_linear_regressor = pickle.load(open(model_filename, "rb"))
        time_input = input_df[["team1_players_alive", "team2_players_alive"]]
        time = time_linear_regressor.predict(time_input)[0]

    input_df["Time"] = time

    # add pistol round marker
    input_df["pistol_round"] = [1 if ((i == 1) or (i == 9)) else 0 for i in input_df.Round]

    # add score
    rounds_ = get_score(hsv_img)

    input_df["team1_score"] = rounds_.count("t")
    input_df["team2_score"] = rounds_.count("ct")

    # add consec wins
    # reduce score list to rounds since last pistol round
    if input_df["Round"].item() > 8:
        # reset the score back
        rounds_ = rounds_[8:]

    rounds_ = np.array(rounds_)
    # find consecutive round wins for both teams
    if len(rounds_) == 0:
        team1_consec_wins = 0
        team2_consec_wins = 0
    elif rounds_[-1] == "t":
        team2_consec_wins = 0
        np.place(rounds_, rounds_ == "t", 1)
        np.place(rounds_, rounds_ == "ct", 0)
        team1_consec_wins = consecutive_binary_counter(rounds_.astype(int)).tail(1).item() + 1
    else:
        team1_consec_wins = 0
        np.place(rounds_, rounds_ == "t", 0)
        np.place(rounds_, rounds_ == "ct", 1)
        team2_consec_wins = consecutive_binary_counter(rounds_.astype(int)).tail(1).item() + 1

    input_df["team1_consec_wins"] = team1_consec_wins
    input_df["team2_consec_wins"] = team2_consec_wins

    # need to change the order of columns to make sure PCA happens in correct order
    # do this by first loading in the features_df
    column_order = pd.read_pickle("features_dfs/features_df.pkl").drop(["match_ID", "team2_won_round", "team1_won_round"], axis=1).columns
    input_df = input_df.reindex(columns=column_order)

    return input_df
