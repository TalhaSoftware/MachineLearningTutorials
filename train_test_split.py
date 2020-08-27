# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 11:05:25 2020

@author: Talha Yazilim
"""

import pandas as pd
import numpy as np

veriler = pd.read_csv("Real Estate.csv")

X = veriler.iloc[:,1:7].values
y = veriler.iloc[:,7].values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=0)
