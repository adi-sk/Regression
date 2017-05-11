# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups_MultipleLinearRegression.csv')
X = dataset.iloc[:, :-1].values # Indepentent Variables R&D spend,Administration,Marketing spend,state
y = dataset.iloc[:, 4].values # dependent variables Profit

                
# Encoding categorical variables from data set we can see that there is only one column which is of independent variable 'states' has to be encoded
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
                
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])

#In this case it is good to encode categorical variables to specific mathematical no. but it should not happen that machine would compare this categories according to the ascending or descending order of the no,. is given
# to avoid this we will use dummy variables which will include column for each category and if the value is equal to that category we will make that specific row '1'
onehotencoder = OneHotEncoder(categorical_features= [3]) #onehotencoder actually creates object to convert categorical variable into columns
X = onehotencoder.fit_transform(X).toarray(); # simply fitting data into object
                
# Avoiding the dummy variable trap
# Mostly python provides some libraries which take care of dummy varible trap

X = X[:,1:] # We are just avoiding dummy variable trap here basically what we are doing is we are avoid one of the dummy variable to be in equation                               
                
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling   # library is going to take care of feature scaling 
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

                            
# Fitting Multiple linear regression to training set                            
# we are going to import the same class what we did in simple linear regression but the difference here will be we will create linear regression model for every independent variable present here

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# predicting Test set result
y_pred = regressor.predict(X_test)

