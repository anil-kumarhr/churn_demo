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
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
import pickle

churn = pd.read_excel(r"CHURNDATA.xlsx")
churn2=churn[['# total debit transactions for S3', 'total transactions',
       'total debit transactions', '# total credit transactions for S3',
       '# total debit transactions for S2',
       '# total debit transactions for S1', 'total credit transactions',
       '# total credit transactions for S2',
       '# total credit transactions for S1','Status']]

label_encoder = preprocessing.LabelEncoder()
churn2['Status']= label_encoder.fit_transform(churn2['Status']) 
pd.to_numeric(churn2['Status'])




class_1 = churn2[churn2['Status'] == 1]
class_0 = churn2[churn2['Status'] == 0]
print(class_0.shape)
class_1_under = class_0.sample(800)

test_under = pd.concat([class_1_under, class_1], axis=0)

test_under = pd.concat([class_1_under, class_1], axis=0)
a = test_under.iloc[:,0:9]
b = test_under.iloc[:,9:]

usmote = SMOTE(random_state=10)
X_smote, y_smote = usmote.fit_resample(a,b)
print(X_smote.shape)


regressor =  RandomForestClassifier( n_estimators= 30, min_samples_leaf= 8, max_depth=8)

#Fitting model with trainig data
regressor.fit(X_smote, y_smote)
Ypred=regressor.predict(X_smote)
print(classification_report(y_smote, Ypred))

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
