# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('polynomial_regression_Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values # always make sure that independent variables are in matrix that is why 1:2
y = dataset.iloc[:, 2].values   #and Dependent varible is in vector

# Splitting the dataset into the Training set and Test set   # no need to split dataset in training set test set bcz we have only 10 observations here
'''from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)'''

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Fitting Linear Regression to dataset      # this regressor is created to compare the result between linear regression model and Polynomial Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)
 
# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)    #this poly_reg object is a transformer tool that is going to transfer matrix of X into new matrix X  which will contain columns of X,X^2,X^3..... here as we used degree = 2 it will create X and X^2 this object will also contain one more column which will have only 1s which will take care of that constant variable in expression
X_poly = poly_reg.fit_transform(X)      
    #now we will fit this X_poly matrix into the linear regression model 
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)        