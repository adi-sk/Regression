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

# Building the optimal model using Backward Ellimination
  # the goal in this method is to select highly statistically significant that has great impact on dependent variable and to remove those which are not impacting dependent varible   
  
    #library which we use here for Backward Elimination
import statsmodels.formula.api as sm # this library is used to test the statistical significance of variable

# NOTE : as we are using this library one thing should be taken care of that this library will not add constant value to our eqn that is if eqn is y(independent VAR)= C + c1x1+c2x2+.....+cnxn  here this library will not take care of value 'C' 
# so what we can do here to Add C is we will add x0=1 for will make eqn  y = c0x0+c1x1+c2x2+.....+cnxn  now here (C = 1)

X = np.append(arr = np.ones((50,1)).astype(int), values = X,axis = 1) # adding extra column to independent data set with all 1s

# now we will create optimal matrix of feature which will include highly statistically significant independent Variables which has high impact on profits          
                # step 1 : add all the Variables to optimal matrix
X_opt = X[:,[0,1,2,3,4,5]]

                 #step 2 : fit full model with all possible predictors 
regressor_OLS = sm.OLS(endog = y , exog = X_opt).fit()#if we open documentation of OLS class it is clearly mention that intercept is not included has to be included by user that is what we did on line 62
regressor_OLS.summary() #this will help us to find the p-value of every variable , lesser is the p-value more is the indepent value impacts the dependent variable  

                     
                 # step 3 : remove the variable with highest p-value and fit the model again without that variable (here to select high and low P-value we have considered significant value = 0.05 if p-value > 0.5 it will get removed from the optimal matrix)    
X_opt = X[:,[0,1,3,4,5]]  # second variable deleted it is having p-value greter than 0.05                     
regressor_OLS = sm.OLS(endog = y , exog = X_opt).fit()# again fit 
regressor_OLS.summary()   

# repeat these steps till we get highest p-value < 0.05
                  
X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog = y , exog = X_opt).fit()# again fit 
regressor_OLS.summary()

X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = y , exog = X_opt).fit()# again fit 
regressor_OLS.summary()

X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog = y , exog = X_opt).fit()# again fit 
regressor_OLS.summary()                        
