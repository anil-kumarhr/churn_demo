# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 22:17:01 2022

@author: anilhr
"""

import flask
from flask import Flask, request , jsonify, render_template
import numpy as np
import pandas as pd
import pickle


app= Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))



@app.route('/')
def man():
    return render_template('home.html')




@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    arr = np.array([[data1, data2, data3]])
    pred = model.predict(arr)
    return render_template('after.html',data=pred)


if __name__ == "__main__":
    app.run(debug=True)
    
