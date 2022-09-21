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
    data4 = request.form['d']
    data5 = request.form['e']
    data6 = request.form['f']
    data7 = request.form['g']
    data8 = request.form['h']
    data9 = request.form['i']
    arr = np.array([[data1, data2, data3, data4, data5, data6, data7, data8, data9]])
    pred = model.predict(arr)
    return render_template('after.html',data=pred)


if __name__ == "__main__":
    app.run(debug=True)
    
