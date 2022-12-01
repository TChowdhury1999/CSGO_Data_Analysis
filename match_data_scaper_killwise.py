# -*- coding: utf-8 -*-
"""
Created on Sun 9 Oct 2022
Author: Tanjim Chowdhury

Scrapes csgo.stats for game information at the kill level and collects info
that is available to the player such as each players K/A/Ds, game score, player
count & the time.

"""


import pandas as pd
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
            team1_won_round.append(team1_score)
            team2_won_round.append(team2_score)
            team1_rounds_won.append(team1_rounds_won[-1] + team1_score)
            team2_rounds_won.append(team2_rounds_won[-1] + team2_score)

        # set winner of game columns
        # count draw as win

        if team1_score > team2_score:
            team1_won_game = [1] * len(team1_score)
            team2_won_game = [0] * len(team2_score)
        elif team1_score == team2_score:
            team1_won_game = [1] * len(team1_score)
            team2_won_game = [1] * len(team2_score)
        else:
            team1_won_game = [0] * len(team1_score)
            team2_won_game = [1] * len(team2_score)

        # populate player DataFrame

        round_list = list(soup.findAll(attrs={"class": "round-info-side"}))[1::2]

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
                kill_time_list = re.findall(r"\d{2}", kill_comp[0])[0:2]
                kill_time = int(kill_time_list[0]) * 60 + int(kill_time_list[1])
                time_.append(kill_time)

                # obtain killer and add kill to player df
                killer_index = player_names[kill_comp[1]] - 1
                player_df.iat[0, killer_index] = player_df.iat[0, killer_index].append(1)
                # add a 0 to the kill list of all the other players
                remaining_indices = list(range(1,11))
                #del remaining_indices[killer_index]

                # check if assist
                # assists have a plus as third component in the kill tag

                # if assist, add an assist to player_df
                if re.search(r"\+", kill_comp[2]):
                    # there is an assist by player in 4th component
                    # issue is that theres weird white space before the player's name
                    # just remove white space and try match
                    # add an assist to the assist column
                    assister_index = player_names[kill_comp[3].lstrip()] - 1
                    player_df.iat[1, assister_index] = player_df.iat[1, assister_index].append(1)
                else:
                    assister_index = None
                
                # the player that died is always the last kill component
                dead_index = player_names[kill_comp[-1]] - 1
                player_df.iat[1, assister_index] = player_df.iat[1, assister_index].append(1)
                
                


        # begin with the rolling score
        # search for the class
        rows = soup.findAll(attrs={"class": "round-score"})
        row_list = list(rows)

        for row in row_list:

            # get the numbers (score) from the text of the row (and conv str to int)
            score = [int(i) for i in row.text if i.isdigit()]
            team1_rounds_won.append(score[0])
            team2_rounds_won.append(score[1])

        # now players alive
        rows = soup.findAll(attrs={"class": "round-outer"})
        row_list = list(rows)

        for row in row_list:

            # list with elements indicating if a player survived or not
            indicators = list(row.find_all(class_="player-indicator"))
            # convert these tags to strings for easier differentiating
            indicators = [str(tag) for tag in indicators]

            # count how many players survived for both teams

        # now equipment val, cash and cash spent
        # as well as time of kill

        rows = soup.findAll(attrs={"class": "round-info-side"})
        row_list = list(rows)

        for row in row_list[::2]:
            # only get the left side of the info page
            # ie the part that stores monetary values

            monetary_vals = re.findall(r"\d+", row.text)
            # conv strs to ints
            monetary_vals = [int(i) for i in monetary_vals]

        for _round in row_list[1::2]:
            # only get right hand side now
            # this has kill times, guns used and player names but only use times for
            # now
            # assume team1 is t side for whole game, afterwards calc half way point
            # and swap the times after that point

            # store each rounds kill times in these lists
            t_side = []
            ct_side = []

            # each kill in this round has a tag containing the team and the time
            kill_tags = _round.findAll(attrs={"class": "tl-inner"})

            for kill in range(len(kill_tags)):
                # loop through each kill and obtain kill time

                # in the kill tag, the time is text on 2nd line in format XX:XX
                kill_time_str = list(kill_tags[kill])[1].text
                kill_time_lst = re.findall(r"\d{2}", kill_time_str)
                kill_time_lst = [int(i) for i in kill_time_lst]

                # time is 60*minutes + secs
                kill_time = kill_time_lst[0] * 60 + kill_time_lst[1]

                # now find the team to add the kill to

                if not re.findall(r"team-ct", str(list(kill_tags[kill])[3])):
                    # lot going on here
                    # in the kill tag, the third line contains either team-ct or team-t
                    # try to find this part of the string and store in list.
                    # if the list is empty, this boolean returns true, so true = team T

                    # team-T got kill if we get here
                    t_side.append(kill_time)
                else:
                    # team-CT got kill if we get here
                    ct_side.append(kill_time)

            # once all kills in round are done, add the lists to their corresponding
            # DF colmns

        # now that columns are complete, compile into a DF

        # now fix timings

        # =============================================================================
        #         second_pistol_rnd = max([team1_rounds_won[-1], team2_rounds_won[-1]])
        #         # the first round of second half is equal to 9 or 16 for short/long matches
        #         # these vals are same as winning amount of rounds for those matches
        #
        #         # store 2nd half kills of team2 (acc team1s times) in temp file
        #         temp_ = team2_kill_times[second_pistol_rnd - 1 :]
        #         # change 2nd half kills of team2 to correct time
        #         match_df["team2_kill_times"] = (
        #             team2_kill_times[: second_pistol_rnd - 1] + team1_kill_times[second_pistol_rnd - 1 :]
        #         )
        #         match_df["team1_kill_times"] = team1_kill_times[: second_pistol_rnd - 1] + temp_
        # =============================================================================

        # now save this dataframe with the filename as the match ID
        # match_df.to_pickle(f"match_dfs/{match_ID}")

        # print msg saying it has been saved and implement a time delay to prevent
        # being banned from the website
        print(f"Saved Match {match_ID}")
        time.sleep(10)
