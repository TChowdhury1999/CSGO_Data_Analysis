# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 22:33:01 2022

@author: Tanjim

Read in match_dfs and collate into a single dataframe

Then perform analysis on the main dataframe
"""


import numpy as np
import pandas as pd
import os.path
import plotly.io as pio
import plotly.express as px
pio.renderers.default='browser'



###############################################################################
# MAIN DF GENERATION
###############################################################################

# load in match list and calc its length which is used by many functions below
match_id_list = list(pd.read_csv("match_IDs.csv")["0"])
no_matches = 48
# no_matches = len(match_id_list)



def init_main_df(match_id_list = match_id_list):
    # initialise main dataframe here

    # load in the first match from the match list
    # add columns that label each round with the match ID, round
    # number
    
    # then return the df and this is the start of the main_df
    
    if os.path.exists(f"match_dfs/{match_id_list[0]}"):
        
        first_df = pd.read_pickle(f"match_dfs/{match_id_list[0]}")
        
        match_id_col = [match_id_list[0]]*len(first_df)
        round_col = np.arange(1, len(first_df)+1)

        first_df["match_id"] = match_id_col
        first_df["round"] = round_col
        
        return(first_df)
    
    else:
        raise FileNotFoundError('First match in match ID list not found')
        
def add_round_winner(df):
    """ Adds columns to df which indicates if team1 or team2 won the round
    """
    # perform inverse cumsum on team1_rounds_won but hold onto all index 0 (round 1)
    # scores and restore after

    first_round_scores = df[df["round"]==1]["team1_rounds_won"]

    team1_win = df["team1_rounds_won"]
    team1_win_arr = np.array(team1_win)
    team1_win_arr[1:] -= team1_win_arr[:-1]
    team1_win = pd.Series(team1_win_arr)

    team1_win.loc[first_round_scores.index] = first_round_scores
            
    df["team1_win"] = team1_win
    
    # team2_win is just the boolean inverse
    
    df["team2_win"] = 1 - team1_win
    
    return df
    
def add_pistol_round(main_df):
    """ Adds a column which indicates if a round is a pistol round or not 
        Also adds column indicating if a short/long match is played
    """
    
    # first need a column that indicates long/short match
    # found by looking at how many rounds the winning team has
    
    group_match_id = main_df.groupby("match_id").max()
    max_round = group_match_id[["team1_rounds_won", "team2_rounds_won"]].max(axis=1)
    max_round_repeated = list(max_round.repeat(group_match_id["round"]))
    
    main_df["last_round"] = max_round_repeated    
    
    # now sum the score each round
    # for 8/9 rounds to win -> round with sum total 9 is pistol round (and 1)
    # for 15/16 rounds to win -> round with sum total 16 is pistol round (and 1)
    
    main_df["pistol_round"] = 0
    
    main_df.loc[main_df["round"] == 1, "pistol_round"] = 1 
    
    main_df["score_sum"] = main_df["team1_rounds_won"] + main_df["team2_rounds_won"]

    main_df.loc[((main_df["last_round"] == 8) | (main_df["last_round"] == 9)) & (main_df["score_sum"] == 9), "pistol_round"] = 1 
    
    # can now remove score_sum
    
    del main_df["score_sum"]
    
    return main_df
    
    

def gen_main_df(no_matches=no_matches, match_id_list=match_id_list):
    
    """ Generates the main df that holds information for matches in the match
        list
        no_matches specifies how many matches in the match list should be 
        included
    """
    
    # check valid input
    if no_matches > len(match_id_list):
        raise IndexError("Maximum number of matches for the list provided exceeded")
    
    # first initialise the main_df
    main_df = init_main_df()

    # generate the rest of the main dataframe by loading in the rest of the 
    # matches and remembering first match is already loaded into the df
    
    # perform the same operations to add match_id and round number columns
    # then concat to the main df
    
    for match in match_id_list[1:no_matches]:
        
        if os.path.exists(f"match_dfs/{match}"):
            
            current_df = pd.read_pickle(f"match_dfs/{match}")
            
            match_id_col = [match]*len(current_df)
            round_col = np.arange(1, len(current_df)+1)

            current_df["match_id"] = match_id_col
            current_df["round"] = round_col

            main_df = pd.concat([main_df, current_df])

        else:
            raise FileNotFoundError(f"The DataFrame of match with ID:{match} was not found")
        
        
    # now reset the dataframe index as we have the round colmn
    main_df.reset_index(drop=True, inplace=True)
                
    # add round winner columns
    main_df = add_round_winner(main_df)
    
    # add pistol round indicator column
    
    
    # some columns aren't int type 
    # change here
    colmns = list(main_df.columns)
    colmns.remove("team1_kill_times")
    colmns.remove("team2_kill_times")
    
    for colmn in colmns:
        main_df[colmn] = pd.to_numeric(main_df[colmn])
    
    return main_df


# generate main df
main_df = gen_main_df()

###############################################################################
# ANALYSIS
###############################################################################

# functions here generate interesting graphs

def win_perc_equip_val(bin_no = 15, pistol=False):
    
    """ Returns plot comparing equipment value to round win percentage,
        bin_no -> number of bins for equipment value
        pistol -> if True, only uses pistol rounds
    """
    
    
    if not pistol:
        # dont need to differentiate between teams so combine them
        
        equip_val = list(main_df["team1_equipment_value"]) + list(main_df["team2_equipment_value"])
        round_win = list(main_df["team1_win"]) + list(main_df["team2_win"])
        
        df_ = pd.DataFrame()
        df_["equip_val"] = equip_val
        df_["round_win"] = round_win
        
        # group equipment value into brackets
        # take a mean of the team1_win colmn to give win perc
        
        grouped_equipment_val = df_.groupby(pd.cut(df_["equip_val"], bin_no)).mean()
    
    
        fig = px.scatter(grouped_equipment_val, x="equip_val", y="round_win", trendline="ols")
        fig.show()
        
    else:
        # add a column indicating if a round is pistol round
        # first need a column indicating if match was short/long match
        























