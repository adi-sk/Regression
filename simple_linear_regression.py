# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values  # here X represents Experience which is an independent varialble
y = dataset.iloc[:, 1].values    # Y represents Salary which is an independent variables 

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling      # We do not need feature scaling here bcz the library that we are going to use for 'Simple Linear regression' will take care of that
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""
#------------------------------------------#-------------------------------------------------------#


                            #Fitting Simple linear regression to the Training set
                            
# here by using sci-kit learn which linear_model library it is really easy to apply simple linear regression on data 

from sklearn.linear_model import LinearRegression  # from this library we are going to import LinearRegression Class
regressor = LinearRegression()
regressor.fit(X_train,y_train) #this is a step where machine actually learnt relation between independent and dependent vectors


#-------------------------------------------#-------------------------------------------------------#

                            #Predicting the test set results 
                            
y_pred = regressor.predict(X_test)        

#-----------------------------------------#------------------------------------------------------#

                            #Visualising the Training Set results
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train , regressor.predict(X_train),color='blue')
plt.title('Salary Vs Experience(Training Set)')
plt.xlabel('Experience in years')
plt.ylabel('Salaries in $')
plt.show()

                            #Visualising the Test Set results
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train , regressor.predict(X_train),color='blue')  # even if we use test set here we will get the same regression line bcz machine has learnt some linear equation on this data
plt.title('Salary Vs Experience(Test Set)')
plt.xlabel('Experience in years')
plt.ylabel('Salaries in $')
plt.show()
