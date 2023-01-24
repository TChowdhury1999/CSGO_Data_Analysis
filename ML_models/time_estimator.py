# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 12:57:17 2023

@author: Tanjim

This scripts creates a quick linear regression model to predict time based on 
the other features as time is only known to the player sometimes (before bomb
                                                                  plant)
"""


import pickle
import pandas as pd
import git
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# load in the features dataframe
repo = git.Repo(".", search_parent_directories=True)
data = pd.read_pickle(repo.working_tree_dir + "/features_dfs/features_df.pkl")

# Split the data set into training and test splits (90/10 split used)
x_train, x_test, y_train, y_test = train_test_split(
    data.drop(["team1_won_round","team1_won_game"], axis=1), data.team1_won_round, test_size=0.10, random_state=0
)

# Make linear regression model instance
linear_regression = LinearRegression()

# fit to the data
linear_regression.fit(x_train, y_train)

# score model
score = linear_regression.score(x_test, y_test)

# save the model
filename = "time_estimator.sav"
pickle.dump(linear_regression, open(filename, "wb"))

##############################################################################

time = get_time(hsv_img)

if len(time) == 0:
    # load linear regression time model
    model_filename = "time_estimator.sav"
    time_linear_regressor = pickle.load(open(filename, 'rb'))
    time_input = read_df.drop["Time"]   # check input matches
    time = time_linear_regressor.predict(time_input)
else:
    # insert time parse from min:sec to sec
    # assume here that its "M:SS"
    time_list = re.findall(r"\d{2}", time)
    time = int(time_list[0]) * 60 + int(time_list[1])

# create input df that will have the features ready for the PCA model
input_df = read_df[
    ["Round", "Time"]
].copy()
working_df = pd.DataFrame()

for player_number in range(1, 11):

    # rounds
    working_df[f"player{player_number}_k_per_round"] = (
        read_df[f"player{player_number}_kills"] / read_df["Round"]
    )
    read_df[f"player{player_number}_a_per_round"] = (
        read_df[f"player{player_number}_assists"] / read_df["Round"]
    )
    read_df[f"player{player_number}_d_per_round"] = (
        read_df[f"player{player_number}_deaths"] / read_df["Round"]
    )

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
    