
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv("weight-height.csv")

X = veriler.iloc[:,1:3].values
y = veriler.iloc[:,0].values

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=0)

from sklearn.linear_model import  LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)

sonuc = lr.predict(np.array([73,180]).reshape(1,-1))

if sonuc > 0.5:
    print("Erkektir ")
elif sonuc < 0.5:
    print("Kadındır ")
    
def own_r2score(y_test,y_hat):

    SS_Total = sum((y_hat-y_test)**2)

    SS_Residual = sum((y_test-np.mean(y_test))**2)

    r2SCORE = 1 - float((SS_Total/SS_Residual))

    return r2SCORE


y_hat = lr.predict(x_test)
r2 = own_r2score(y_test,y_hat)

print(r2)

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=4)
x_poly = poly.fit_transform(x_train)
poly_lr = LinearRegression()
poly_lr.fit(x_poly,y_train)


sonuc2 = poly_lr.predict(poly.fit_transform(np.array([65,100]).reshape(1,-1)))

if sonuc2 > 0.5:
    print("Erkektir ")
elif sonuc2 < 0.5:
    print("Kadındır ")

def own_r2score(y_test,y_hat):

    SS_Total = sum((y_hat-y_test)**2)

    SS_Residual = sum((y_test-np.mean(y_test))**2)

    r2SCORE = 1 - float((SS_Total/SS_Residual))

    return r2SCORE


y_hat = poly_lr.predict(poly.fit_transform(x_test))
r2 = own_r2score(y_test,y_hat)

print(r2)


