# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 10:10:36 2023

@author: tchowdh

Flask web application that will display results of predictor


Used https://blog.miguelgrinberg.com/post/dynamically-update-your-flask-web-pages-using-turbo-flask
for turbo-flask implementation


"""

from flask import Flask, render_template
import random



app = Flask(__name__)



@app.route('/')
def main_page():
    return render_template('main_page.html')

@app.context_processor
def inject_winner_and_probability():
    winner = random.choice(["team1", "team2"])
    probability = random.random()*50+50
    return {"winner":winner,"probability":probability}
