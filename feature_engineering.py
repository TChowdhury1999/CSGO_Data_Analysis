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


def consecutive_binary_counter(series):

    # returns a series where each row has the consecutive number of 1s previously since the last 0 in the input series
    # so [1,0,0,1,1,1,0,1,1] outputs [0,1,0,0,1,2,3,0,1]
    series = pd.Series(series)
    return series.groupby(series.ne(series.shift()).cumsum()).cumsum().shift().fillna(0).astype(int)


if __name__ == "__main__":

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

    for metric in ["k", "d", "a"]:
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
            features_df[f"team1_min_{metric}_per_{round_ind}r_of_alive_p"] = (
                working_df[[f"player{player_number}_{metric}_per_{round_ind}round" for player_number in range(1, 6)]]
                .multiply(np.array(combined_df[[f"player{player_number}_alive" for player_number in range(1, 6)]]))
                .min(axis=1)
            )
            features_df[f"team1_mean_{metric}_per_{round_ind}r_of_alive_p"] = (
                working_df[[f"player{player_number}_{metric}_per_{round_ind}round" for player_number in range(1, 6)]]
                .multiply(np.array(combined_df[[f"player{player_number}_alive" for player_number in range(1, 6)]]))
                .mean(axis=1)
            )
            features_df[f"team1_max_{metric}_per_{round_ind}r_of_alive_p"] = (
                working_df[[f"player{player_number}_{metric}_per_{round_ind}round" for player_number in range(1, 6)]]
                .multiply(np.array(combined_df[[f"player{player_number}_alive" for player_number in range(1, 6)]]))
                .max(axis=1)
            )

            # team 2
            features_df[f"team2_min_{metric}_per_{round_ind}r_of_alive_p"] = (
                working_df[[f"player{player_number}_{metric}_per_{round_ind}round" for player_number in range(6, 11)]]
                .multiply(np.array(combined_df[[f"player{player_number}_alive" for player_number in range(6, 11)]]))
                .min(axis=1)
            )
            features_df[f"team2_mean_{metric}_per_{round_ind}r_of_alive_p"] = (
                working_df[[f"player{player_number}_{metric}_per_{round_ind}round" for player_number in range(6, 11)]]
                .multiply(np.array(combined_df[[f"player{player_number}_alive" for player_number in range(6, 11)]]))
                .mean(axis=1)
            )
            features_df[f"team2_max_{metric}_per_{round_ind}r_of_alive_p"] = (
                working_df[[f"player{player_number}_{metric}_per_{round_ind}round" for player_number in range(6, 11)]]
                .multiply(np.array(combined_df[[f"player{player_number}_alive" for player_number in range(6, 11)]]))
                .max(axis=1)
            )

    # add a pistol round marker
    features_df["pistol_round"] = [1 if ((i == 1) or (i == 9)) else 0 for i in features_df.Round]

    # add a number of alive players counter
    features_df["team1_players_alive"] = (
        combined_df.player1_alive
        + combined_df.player2_alive
        + combined_df.player3_alive
        + combined_df.player4_alive
        + combined_df.player5_alive
    )
    features_df["team2_players_alive"] = (
        combined_df.player6_alive
        + combined_df.player7_alive
        + combined_df.player8_alive
        + combined_df.player9_alive
        + combined_df.player10_alive
    )

    # add a consecutive round win/loss since pistol counter

    # create a working dataframe that is grouped by rounds of each match
    working_round_df = (
        features_df.groupby(["match_ID", "Round"])
        .min()[["team1_won_round", "team2_won_round", "pistol_round"]]
        .reset_index(drop=False)
    )

    # we want to apply the consecutive_binary_counter function to sections of
    # "team1_won_round" separated by the presence of a pistol round (that way the
    # counter doesn't go across halves/games). Do this by cumsumming pistol round
    # flag and then groupby by this new column and using:
    # df.groupby('A')['B'].transform('func')

    working_round_df["cumsum_pistol"] = working_round_df.pistol_round.cumsum()

    for team_no in range(1, 3):
        working_round_df[f"team{team_no}_consec_wins"] = working_round_df.groupby("cumsum_pistol")[
            f"team{team_no}_won_round"
        ].apply(consecutive_binary_counter)

    # now join the consec wins columns to the feature df by joining on match ID and
    # round

    features_df = pd.merge(
        features_df,
        working_round_df[["match_ID", "Round", "team1_consec_wins", "team2_consec_wins"]],
        on=["match_ID", "Round"],
        how="left",
    )

    # save the final dataframe
    features_df.to_pickle("features_dfs/features_df.pkl")
