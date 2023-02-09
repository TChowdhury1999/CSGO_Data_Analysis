# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 10:10:36 2023

@author: tchowdh

Flask web application that will display results of predictor

"""

from flask import Flask, render_template




app = Flask(__name__)


@app.route('/')
def main_page():
    return render_template('main_page.html')

