# CSGO_Data_Analysis
Personal project involving webscraping from csgo-stats.gg using it to create live round winning probability overlay ingame


- match_id_scraper.py creates a list of match IDs for the input profile
- match_data_scraper.py then goes through these matches on csgostats.gg and saves the match data into DFs
- match_data_analyser.py is used to produce graphs displaying different characteristics and how they affect round win % for example.
