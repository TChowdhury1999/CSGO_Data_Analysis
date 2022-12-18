# -*- coding: utf-8 -*-
"""
Created on Sun 9 Oct 2022
Author: Tanjim Chowdhury

Scrapes csgo.stats for game information at the kill level and collects info
that is available to the player such as each players K/A/Ds, game score, player
count & the time.

"""


import pandas as pd
import numpy as np
import os.path
import re
import time
import codecs

from selenium import webdriver
from bs4 import BeautifulSoup

# read in html example for WIP
page_html=codecs.open("example_html.html", 'r', 'utf-16')


# define the base URL and extension for matches
# full URL is base_URL+matchID+extension
base_URL = "https://csgostats.gg/match/"
extension = "#/rounds"

# read in match_IDs
match_IDs = list(pd.read_csv("match_IDs.csv")["0"])

# loop through each match ID

for match_ID in match_IDs:

    # check if the match has already been scraped and if so pass to next match
    # in the match list
    if os.path.exists(f"match_dfs_killwise/{match_ID}"):
        pass

    # otherwise scrape the match for data
    else:

        """DataFrame defining"""

        """
        Round DataFrame has information that is constant the whole round
        Format:

        Round|Team 1 Rounds Won|Team 2 Rounds Won|Team 1 Won Round|Team 2 Won Round|Team 1 Won Game|Team 2 Won Game
        -----------------------------------------------------------------------------------------------------------
        1    |0                |0                |True            |False           |False          |True
        .
        .
        5    |1                |3                |False           |True            |False          |True
        """

        # Round DataFrame columns
        round_ = []
        team1_rounds_won = [0]
        team2_rounds_won = [0]
        team1_won_round = []
        team2_won_round = []
        team1_won_game = []
        team2_won_game = []

        """
        Player stats DataFrame contains the k/d/a for each player with the
        format:
        
               |Player 1   |Player 2 ... |Player 9   |Player 10             
        Kills  |[0,0,1,...]|[0,1,0,...]  |[0,0,0,...]|[0,0,0,...]
        Deaths |[0,0,0,...]|[0,0,0,...]  |[0,1,0,...]|[0,0,1,...]
        Assists|[0,0,0,...]|[0,0,1,...]  |[0,0,0,...]|[0,0,0,...]
        
        where each list contains a value for each timestamp
        
        """
        player_df = pd.DataFrame(
            [
                [[] for _ in range(10)],
                [[] for _ in range(10)],
                [[] for _ in range(10)],
            ],
            index=["kills", "deaths", "assists"],
            columns=[[f"player_{x}" for x in range(1, 11)]],
        )

        # timestamps defined below
        time_ = []

        """
        Final dataframe format after collating all the data will be:
        
        Round|Time|Score|player 1|player 2|player 3 ....|Won round|Won game
        --------------------------------------------------------------------
        1    |12  |2-4  |[k,d,a]| [k,d,a] |[k,d,a] ...  | CT      | T
             |45  |2-4  |[k,d,a]| [k,d,a] |[k,d,a] ...  | CT      | T
             
        2    |23  |3-4  |[k,d,a]| [k,d,a] |[k,d,a] ...  | CT      | T
        .
        .
        .
        -------------------------------------------------------------------
        
        """

        """
        Logic:
            1) Fill in round dataframe
            2) a) For each kill, find killer and make list of names that aren't
                  them
               b) Fill in 0 on kill lists on Player stats DataFrame for this 
                  list of names
               c) Fill in 1 for the killer
               d) Repeat for deaths & assists
               e) Record timestamp
           3) Combine into the final DataFrame
        """

        # Loads up webdriver and starts selenium session at the target website
        browser = webdriver.Chrome("chromedriver.exe")
        browser.get(base_URL + str(match_ID) + extension)

        # get the html of the page
        page_html = browser.page_source
        # close browser
        browser.quit()
        # parse the html using bs4
        soup = BeautifulSoup(page_html, "html.parser")

        # start by populating a player_names dict
        rows = soup.findAll(attrs={"class": "match-player"})
        name_list = [tag.text.strip() for tag in list(rows)]
        player_names = {k: v for v, k in enumerate(name_list, start=1)}

        """ Round DataFrame """

        # get list of score
        score_list = list(soup.findAll(attrs={"class": "round-score"}))

        # loop through score_list and save the scores
        # also collect info about who won etc

        for round_ in score_list:
            team1_score, *_, team2_score = round_.text.strip()
            team1_rounds_won.append(int(team1_score))
            team2_rounds_won.append(int(team2_score))

        team1_won_round = np.diff(team1_rounds_won)
        team2_won_round = np.diff(team2_rounds_won)

        # set winner of game columns
        # count draw as separate outcome 2 so can be used fixed later

        team1_final_score = team1_rounds_won.pop()
        team2_final_score = team2_rounds_won.pop()
        
        game_length = len(team1_won_round)
        round_ = np.arange(1, game_length+1)

        if team1_final_score > team2_final_score:
            team1_won_game = [1] * game_length
            team2_won_game = [0] * game_length
        elif team1_final_score == team2_final_score:
            team1_won_game = [2] * game_length
            team2_won_game = [2] * game_length
        else:
            team1_won_game = [0] * game_length
            team2_won_game = [1] * game_length
            
        # collect into a DataFrame just for ease of access
        round_df = pd.DataFrame(data=np.column_stack([round_, team1_rounds_won, team2_rounds_won, team1_won_round,
                                                      team2_won_round, team1_won_game, team2_won_game]),
                                columns=["round", "team1_rounds_won", "team2_rounds_won", "team1_won_round", "team2_won_round",
                                         "team1_won_game", "team2_won_game"])

        """ Player DataFrame"""

        round_list = list(soup.findAll(attrs={"class": "round-info-side"}))[1::2]
        
        round_number = 1

        for round_ in round_list:

            # loop through the rounds

            # create a kill_list listing all the kills in the round
            kill_list = []
            for kill in round_:
                kill_list.append(kill.text)
            kill_list = kill_list[1::2][:-1]

            for kill in kill_list:
                # each component of kill is separated by new line \n
                kill_comp = kill.split("\n")[1:-1]

                # obtain kill time in s and add to time column
                # store time as round number and time in round
                kill_time_list = re.findall(r"\d{2}", kill_comp[0])[0:2]
                kill_time = int(kill_time_list[0]) * 60 + int(kill_time_list[1])
                time_.append((round_number, kill_time))

                # obtain killer and add kill to player df
                # kill_comp[1] is the killer name
                # but if there is an assist kill_comp[1] has a space after
                # use strip to remove this whitespace
                killer_index = player_names[kill_comp[1].strip()] - 1
                player_df.iat[0, killer_index].append(1)
                # add a 0 to the kill list of all the other players
                remaining_indices = list(range(0,10))
                del remaining_indices[killer_index]
                for player_index in remaining_indices:
                    player_df.iat[0, player_index].append(0)
                
                # check if assist
                # assists have a plus as third component in the kill tag
                
                # before 

                # if assist, add an assist to player_df
                if re.search(r"\+", kill_comp[2]):
                    # there is an assist by player in 4th component
                    # issue is that theres weird white space before the player's name
                    # just remove white space and try match
                    # add an assist to the assist column
                    assister_index = player_names[kill_comp[3].lstrip()] - 1
                    player_df.iat[2, assister_index].append(1)
                    
                    # add a 0 to the assist list of all the other players
                    remaining_indices = list(range(0,10))
                    del remaining_indices[assister_index]
                    for player_index in remaining_indices:
                        player_df.iat[2, player_index].append(0)

                else:
                    # there were no assists for this kill so add a 0 to 
                    # assist list
                    player_indices = list(range(0,10))
                    for player_index in player_indices:
                        player_df.iat[2, player_index].append(0)
                        
                        
                # the player that died is always the last kill component
                dead_index = player_names[kill_comp[-1]] - 1
                player_df.iat[1, dead_index].append(1)
                
                # add a 0 to the death list of the rest of the players
                remaining_indices = list(range(0,10))
                del remaining_indices[dead_index]
                for player_index in remaining_indices:
                    player_df.iat[1, player_index].append(0)        
            
            # increase round number that is used in timestamps
            round_number+=1 
        
        """ Final DataFrame """
        
        # compile all the dataframes into the format shown above
        
        
        
        # now save this dataframe with the filename as the match ID
        # match_df.to_pickle(f"match_dfs/{match_ID}")

        # print msg saying it has been saved and implement a time delay to prevent
        # being banned from the website
        print(f"Saved Match {match_ID}")
        time.sleep(10)
