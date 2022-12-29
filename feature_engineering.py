# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 13:52:18 2022

@author: Tanjim
"""

# -*- coding: utf-8 -*-
"""
Created on Mon 26 Dec 2022
Author: Tanjim Chowdhury

Loads in the match DataFrames, combines them and format them so they can train
a ML Model.

Then create additional features that the ML could use

Note: For v1 of this model we will only predict round outcomes then aggregate to game outcome

"""


import pandas as pd
import numpy as np
import os

""" DataFrame Loading & Quick Formatting """


def read_pkl(path):
    """
    Function that reads in the pickle dataframe at path
    This also adds the match ID as a column
    """
    loaded_df = pd.read_pickle(path)
    match_ID = path[10:-7]
    loaded_df["match_ID"] = match_ID
    return loaded_df


# load in all the dataframes into one dataframe

match_df_paths = ["match_dfs/" + f for f in os.listdir("match_dfs/") if f.endswith(".pickle")]
combined_df = pd.concat(map(read_pkl, match_df_paths))

# add in some useful columns that were left out to save memory
combined_df["team2_won_round"] = 1 - combined_df["team1_won_round"]
combined_df["team2_won_game"] = 0.5 * combined_df["team1_won_game"] ** 3 - 1.5 * combined_df["team1_won_game"] + 1

""" Feature Engineering """

# engineer new features that could be calculated by at a random timestamp
# information available is each row but NOT previous rows
# except is how rounds won/score has evolved over rounds

# work in separate dataframes
features_df = combined_df[
    ["match_ID", "Round", "Time", "team1_score", "team2_score", "team1_won_round", "team2_won_round"]
].copy()
working_df = pd.DataFrame()

# first add min/avg/max of (k/d/a) per round/winning round of total/alive players for team1/team2

# calculate and add a k/d/a per round/winning round column for each player to working df
for player_number in range(1, 11):

    # rounds
    working_df[f"player{player_number}_k_per_round"] = (
        combined_df[f"player{player_number}_kills"] / combined_df["Round"]
    )
    working_df[f"player{player_number}_a_per_round"] = (
        combined_df[f"player{player_number}_assists"] / combined_df["Round"]
    )
    working_df[f"player{player_number}_d_per_round"] = (
        combined_df[f"player{player_number}_deaths"] / combined_df["Round"]
    )

    # winning rounds (score)
    if player_number <= 5:
        score_ = combined_df["team1_score"]
    else:
        score_ = combined_df["team2_score"]

    score_[score_ == 0] = 1

    working_df[f"player{player_number}_k_per_w_round"] = combined_df[f"player{player_number}_kills"] / score_
    working_df[f"player{player_number}_a_per_w_round"] = combined_df[f"player{player_number}_assists"] / score_
    working_df[f"player{player_number}_d_per_w_round"] = combined_df[f"player{player_number}_deaths"] / score_


# now add the stats (min/mean/max) for all/alive players for each team

for metric in ["k","d","a"]:
    for round_ind in ["", "w_"]:
        # all players for each team
        # team1
        features_df[f"team1_min_{metric}_per_{round_ind}r_of_total_p"] = working_df[
            [f"player{player_number}_{metric}_per_{round_ind}round" for player_number in range(1, 6)]
        ].min(axis=1)
        features_df[f"team1_mean_{metric}_per_{round_ind}r_of_total_p"] = working_df[
            [f"player{player_number}_{metric}_per_{round_ind}round" for player_number in range(1, 6)]
        ].mean(axis=1)
        features_df[f"team1_max_{metric}_per_{round_ind}r_of_total_p"] = working_df[
            [f"player{player_number}_{metric}_per_{round_ind}round" for player_number in range(1, 6)]
        ].max(axis=1)
        # team2
        features_df[f"team2_min_{metric}_per_{round_ind}r_of_total_p"] = working_df[
            [f"player{player_number}_{metric}_per_{round_ind}round" for player_number in range(6, 11)]
        ].min(axis=1)
        features_df[f"team2_mean_{metric}_per_{round_ind}r_of_total_p"] = working_df[
            [f"player{player_number}_{metric}_per_{round_ind}round" for player_number in range(6, 11)]
        ].mean(axis=1)
        features_df[f"team2_max_{metric}_per_{round_ind}r_of_total_p"] = working_df[
            [f"player{player_number}_{metric}_per_{round_ind}round" for player_number in range(6, 11)]
        ].max(axis=1)
        
        # alive players for each team
        # logic for alive players is to just multiply 
        # team 1
        features_df[f"team1_min_{metric}_per_{round_ind}r_of_alive_p"] = working_df[
            [f"player{player_number}_{metric}_per_{round_ind}round" for player_number in range(1, 6)]
        ].multiply(np.array(combined_df[[f"player{player_number}_alive" for player_number in range(1, 6)]])).min(axis=1)
        features_df[f"team1_mean_{metric}_per_{round_ind}r_of_alive_p"] = working_df[
            [f"player{player_number}_{metric}_per_{round_ind}round" for player_number in range(1, 6)]
        ].multiply(np.array(combined_df[[f"player{player_number}_alive" for player_number in range(1, 6)]])).mean(axis=1)
        features_df[f"team1_max_{metric}_per_{round_ind}r_of_alive_p"] = working_df[
            [f"player{player_number}_{metric}_per_{round_ind}round" for player_number in range(1, 6)]
        ].multiply(np.array(combined_df[[f"player{player_number}_alive" for player_number in range(1, 6)]])).max(axis=1)
        
        # team 2
        features_df[f"team1_min_{metric}_per_{round_ind}r_of_alive_p"] = working_df[
            [f"player{player_number}_{metric}_per_{round_ind}round" for player_number in range(6, 11)]
        ].multiply(np.array(combined_df[[f"player{player_number}_alive" for player_number in range(6, 11)]])).min(axis=1)
        features_df[f"team1_mean_{metric}_per_{round_ind}r_of_alive_p"] = working_df[
            [f"player{player_number}_{metric}_per_{round_ind}round" for player_number in range(6, 11)]
        ].multiply(np.array(combined_df[[f"player{player_number}_alive" for player_number in range(6, 11)]])).mean(axis=1)
        features_df[f"team1_max_{metric}_per_{round_ind}r_of_alive_p"] = working_df[
            [f"player{player_number}_{metric}_per_{round_ind}round" for player_number in range(6, 11)]
        ].multiply(np.array(combined_df[[f"player{player_number}_alive" for player_number in range(6, 11)]])).max(axis=1)


# add a pistol round marker

# add a consecutive round win/loss since pistol counter
