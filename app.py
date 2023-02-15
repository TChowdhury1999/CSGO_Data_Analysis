# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 10:10:36 2023

@author: tchowdh

Flask web application that will display results of predictor

Uses turbo-flask to create websocket so updated probabilities can be pushed
to webapp

"""

from flask import Flask, render_template
import random
import time
from turbo_flask import Turbo
import threading
import path_to_prediction as ptp

app = Flask(__name__)
turbo = Turbo(app)


@app.route('/')
def main_page():
    return render_template('main_page.html')

@app.before_first_request
def before_first_request():
    threading.Thread(target=update_winner).start()

def update_winner():
    with app.app_context():
        while True:
            time.sleep(5)
            turbo.push(turbo.replace(render_template('injected.html'), 'winner_and_probability'))

@app.context_processor
def inject_winner_and_probability():
    image_directory_path = r"C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\csgo\screenshots"
    tree_outcome, Round = ptp.obtain_prediction(image_directory_path)
    if tree_outcome[0]>=tree_outcome[1]:
        winner="CT"
        probability=tree_outcome[0]
    else:
        winner="T"
        probability=tree_outcome[1]
        
    return {"round":Round, "winner":winner,"probability":probability}