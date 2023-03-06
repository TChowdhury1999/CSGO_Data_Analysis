# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 10:10:36 2023

@author: tchowdh

Flask web application that will display results of predictor

Uses turbo-flask to create websocket so updated probabilities can be pushed
to webapp

"""

from flask import Flask, render_template
import time
import git
import pickle
from turbo_flask import Turbo
import threading
from path_to_prediction import write_prediction, get_latest_file, obtain_prediction
import webbrowser
import pandas as pd
from playsound import playsound
import winsound
from pathlib import Path


# initialise web app
app = Flask(__name__)
turbo = Turbo(app)

# turn off pd warnings
pd.options.mode.chained_assignment = None

# load in ML models
repo = git.Repo(".", search_parent_directories=True)
scaler = pickle.load(open(repo.working_tree_dir + "/ML_models/scaler.sav", "rb"))
PCA = pickle.load(open(repo.working_tree_dir + "/ML_models/PCA.sav", "rb"))
xgbTree = pickle.load(open(repo.working_tree_dir + "/ML_models/xgbTree.sav", "rb"))

# set audio paths 
t_side_path = repo.working_tree_dir + "/sounds/t_side.mp3"
t_side_path = Path().cwd() / (repo.working_tree_dir + "/sounds/t_side.mp3")
ct_side_path = Path(repo.working_tree_dir + "/sounds/ct_side.mp3")

# load in image directory path
with open("path.txt", "r") as file:
    image_directory_path = file.read()

# initialise model
previous_file = latest_file = get_latest_file(image_directory_path)
write_prediction("None")
print("ML Model Initialised")


def update_prediction(image_directory_path, previous_file, latest_file, PCA, scaler, xgbTree):
    """
    Updates the ML model result in the text file if a new leaderboard image
    file is found
    """

    latest_file = get_latest_file(image_directory_path)

    if latest_file != previous_file:
        previous_file = latest_file
        outcome = obtain_prediction(latest_file, PCA, scaler, xgbTree)
        write_prediction(outcome)
        print()

    else:
        print("No new file")
        pass


@app.route("/")
def main_page():
    return render_template("main_page.html")


@app.before_first_request
def before_first_request():
    threading.Thread(target=update_winner).start()


def update_winner():
    with app.app_context():
        while True:
            time.sleep(5)
            turbo.push(turbo.replace(render_template("injected.html"), "winner_and_probability"))



@app.context_processor
def inject_winner_and_probability():
    """
    Pulls the winner and probability from text file
    """

    update_prediction(image_directory_path, previous_file, latest_file, PCA, scaler, xgbTree)

    dict_keys = ["round", "winner", "probability"]
    dict_output = ["...", "...", "..."]

    with open("winner_cache.pickle", "rb") as file:
        tree_outcome = pickle.load(file)

    if tree_outcome == "None":
        return dict(zip(dict_keys, dict_output))
    else:
        pass

    if tree_outcome[0] >= tree_outcome[1]:
        dict_output[1] = "T"
        dict_output[2] = tree_outcome[0]
        playsound(r"t_side_path")
    else:
        dict_output[1] = "CT"
        dict_output[2] = tree_outcome[1]
        playsound(ct_side_path)
        
    dict_output[0] = tree_outcome[2]
    dict_output[2] = int(dict_output[2] * 100)

    return dict(zip(dict_keys, dict_output))


if __name__ == "__main__":
    webbrowser.open_new("http://localhost:5000")
    app.run(host="0.0.0.0", port=5000)
    # pass
