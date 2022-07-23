"""
Created on Fri Jun 24 22:53:48 2022

@author: Tanjim

Scrapes the website: https://csgostats.gg/player/{player_id}/matches for their 
match numbers uploaded to csgo stats

Ignores matches where a player has been banned and maps other than de_mirage

Saves match IDs as a csv file in current directory as match_IDs.csv
"""


import pandas as pd
import selenium 

from selenium import webdriver
from bs4 import BeautifulSoup




# my player ID on steam, can be replaced
player_ID = 76561198312168152
URL = f"https://csgostats.gg/player/{player_ID}?type=comp&maps%5B%5D=de_mirage&date_start=&date_end=#/matches"


# function to extract match IDs

def get_match_numbers(URL = URL):
    
    browser = webdriver.Chrome("chromedriver.exe")
    browser.get(URL)
    # Loads up webdriver and starts selenium session

    # list of match numbers stored here
    match_number_list = []
     
    # get the html of the page
    page_html = browser.page_source
    
    # close browser
    browser.quit()
    
    # parse the html using bs4 
    soup = BeautifulSoup(page_html, 'html.parser')
    
    # search for the class p-row js-link which is each row in the matches table
    rows = soup.findAll(attrs= {"class":"p-row js-link"})
    row_list = list(rows)
    
    # loop through the rows
    for row in row_list:
        string = str(row)
        # all match codes follow format "match/XXXXXXXX"
        # although some are 7 figures not 8
        # first find "/match/"
        start_index = string.find("/match/")
        
        # now save the 7 or 8 letter substring (the match ID) 7 indices after 
        # the start index, as an integer in match_number_list
        
        if string[start_index+15] == "'":
            match_ID = int( string[start_index+7: start_index+15] )
        else:
            match_ID = int( string[start_index+7: start_index+14] )
            
        match_number_list.append(match_ID)
    
    # return list of match IDs
    return match_number_list
        



match_IDs = get_match_numbers()
        
df_match_IDs = pd.Series(match_IDs)
df_match_IDs.to_csv("match_IDs.csv")
        
        
        

        