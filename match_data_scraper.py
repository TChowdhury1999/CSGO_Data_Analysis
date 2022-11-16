# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 14:00:38 2022

@author: Tanjim

Scrapes through multiple matches as listed by their match IDs in match_IDs.csv
and collects data for each round such as round winner, players alive, cash
spent etc, and saves each match as a dataframe in /match_dfs.

"""

import pandas as pd
import numpy as np
import os.path
import selenium
import re
import time

from selenium import webdriver
from bs4 import BeautifulSoup


# the base URL and extension for matches
# full URL is base_URL+matchID+extension
base_URL = "https://csgostats.gg/match/"
extension = "#/rounds"

# read in match_IDs
match_IDs = list(pd.read_csv("match_IDs.csv")["0"])


# loop through each match ID

for match_ID in match_IDs:

    # check if the match has already been scraped
    if os.path.exists(f"match_dfs/{match_ID}"):
        pass
    else:

        browser = webdriver.Chrome("chromedriver.exe")
        browser.get(base_URL + str(match_ID) + extension)
        # Loads up webdriver and starts selenium session at the target website

        # DataFrame columns will be stored below and populated
        # DataFrame rows are the rounds in a game

        team1_rounds_won = []
        team1_players_alive = []
        team1_equipment_value = []
        team1_cash = []
        team1_cash_spent = []
        team1_kill_times = []

        team2_rounds_won = []
        team2_players_alive = []
        team2_equipment_value = []
        team2_cash = []
        team2_cash_spent = []
        team2_kill_times = []

    # get the html of the page
    page_html = browser.page_source

    # close browser
    browser.quit()

    # parse the html using bs4
    soup = BeautifulSoup(page_html, "html.parser")

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
        team1_players_alive.append(indicators[:5].count('<div class="player-indicator survived"></div>'))
        team2_players_alive.append(indicators[5:].count('<div class="player-indicator survived"></div>'))

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

        team1_equipment_value.append(monetary_vals[0])
        team2_equipment_value.append(monetary_vals[1])
        team1_cash.append(monetary_vals[2])
        team2_cash.append(monetary_vals[3])
        team1_cash_spent.append(monetary_vals[4])
        team2_cash_spent.append(monetary_vals[5])

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

        team1_kill_times.append(t_side)
        team2_kill_times.append(ct_side)

    # now that columns are complete, compile into a DF

    column_names = [
        "team1_rounds_won",
        "team2_rounds_won",
        "team1_players_alive",
        "team2_players_alive",
        "team1_cash",
        "team2_cash",
        "team1_cash_spent",
        "team2_cash_spent",
        "team1_equipment_value",
        "team2_equipment_value",
        "team1_kill_times",
        "team2_kill_times",
    ]

    match_df = pd.DataFrame(
        np.array(
            [
                team1_rounds_won,
                team2_rounds_won,
                team1_players_alive,
                team2_players_alive,
                team1_cash,
                team2_cash,
                team1_cash_spent,
                team2_cash_spent,
                team1_equipment_value,
                team2_equipment_value,
                team1_kill_times,
                team2_kill_times,
            ]
        ).transpose(),
        columns=column_names,
    )

    # now fix timings

    second_pistol_rnd = max([team1_rounds_won[-1], team2_rounds_won[-1]])
    # the first round of second half is equal to 9 or 16 for short/long matches
    # these vals are same as winning amount of rounds for those matches

    # store 2nd half kills of team2 (acc team1s times) in temp file
    temp_ = team2_kill_times[second_pistol_rnd - 1 :]
    # change 2nd half kills of team2 to correct time
    match_df["team2_kill_times"] = (
        team2_kill_times[: second_pistol_rnd - 1] + team1_kill_times[second_pistol_rnd - 1 :]
    )
    match_df["team1_kill_times"] = team1_kill_times[: second_pistol_rnd - 1] + temp_

    # now save this dataframe with the filename as the match ID
    match_df.to_pickle(f"match_dfs/{match_ID}")

    # print msg saying it has been saved and implement a time delay to prevent
    # being banned from the website
    print(f"Saved Match {match_ID}")
    time.sleep(1)
