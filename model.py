# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 21:56:38 2022

@author: anilhr
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import pickle

churn = pd.read_excel(r"CHURNDATA.xlsx")
churn=churn[['# total debit transactions for S3','total debit amount for S3','total debit amount','total transactions','total debit transactions','total credit amount for S3','# total debit transactions for S2','AGE','Status']]

label_encoder = preprocessing.LabelEncoder()
churn['Status']= label_encoder.fit_transform(churn['Status']) 
pd.to_numeric(churn['Status'])




class_count_1, class_count_0 = churn['Status'].value_counts()
print(class_count_0)
class_1 = churn[churn['Status'] == 0]
print(class_1)
class_0 = churn[churn['Status'] == 1]
class_1_under = class_1.sample(class_count_0)

test_under = pd.concat([class_1_under, class_0], axis=0)
x = test_under.iloc[:,0:3]
y = test_under.iloc[:,8:]




regressor =  RandomForestClassifier(class_weight={0:1, 1:1}, n_estimators= 80, min_samples_leaf= 1, max_depth=4)

#Fitting model with trainig data
regressor.fit(x, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
