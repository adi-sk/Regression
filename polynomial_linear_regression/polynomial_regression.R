# Polynomial Regression

# Importing the dataset
dataset = read.csv('polynomial_regression_Position_Salaries.csv')
dataset = dataset[2:3] #index in r starts from '1'

# Splitting the dataset into the Training set and Test set  # as the data has only 10 observation we don not need to split data in training set and test set 
# install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$DependentVariable, SplitRatio = 0.8)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling   # feature scaling is also not required here 
# training_set = scale(training_set)
# test_set = scale(test_set)

# fitting Liner regression to the dataset 
lin_reg = lm(formula = Salary ~ Level,
             data = dataset)

# Fitting Polynonmial Regression to dataset
  # here we are going to add polynomial Features, they additional independent variables which are in this case 'Level^2 , Level^3, Level^4 .....'
  # in short what we can conclude from this is polynomial regression module is nothing but Multiple Linear regression model that is composed of one independent variable and additional independent variables that are the polynomial terms of the first independent variable
dataset$Level2 = dataset$Level^2  # this is a polynomial feature or we can say we are adding one extra column here which is the power 2 of Level
dataset$Level3 = dataset$Level^3
poly_reg = lm(formula = Salary ~ . ,
              data = dataset)