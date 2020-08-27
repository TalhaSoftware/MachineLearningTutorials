# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 12:29:25 2020

@author: Talha Yazilim
"""

import pandas as pd
import numpy as np

veriler = pd.read_csv("Real Estate.csv")

X = veriler.iloc[:,1:7].values
y = veriler.iloc[:,7].values


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=0)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)

from sklearn.metrics import r2_score
y_hat = lr.predict(x_test)
print(r2_score(y_test,y_hat))

def own_r2score(y_test,y_hat):
    SS_Total = sum((y_hat-y_test)**2)
    SS_Residual = sum((y_test-np.mean(y_test))**2)
    r2SCORE = 1 - float((SS_Total/SS_Residual))
    return r2SCORE

r2 = own_r2score(y_test,y_hat)
print(r2)

print(lr.predict(np.array([2013,10,500,5,24.945632,121.542353]).reshape(1,-1)))


