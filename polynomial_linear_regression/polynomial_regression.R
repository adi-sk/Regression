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
dataset$Level4 = dataset$Level^4
poly_reg = lm(formula = Salary ~ . ,
              data = dataset)

#Visualizing the linear regression Results
ggplot()+
  geom_point(aes(x= dataset$Level , y = dataset$Salary),
             color = 'red')+
  geom_line(aes(x= dataset$Level , y = predict(lin_reg,newdata = dataset)),
            color = 'blue')+
  ggtitle('truth or bluff (Linear Regression)')+
  xlab('Level')+
  ylab('Salary')
  
#Visualizing the polynomial Regression results
ggplot()+
  geom_point(aes(x= dataset$Level , y = dataset$Salary),
             color = 'red')+
  geom_line(aes(x= dataset$Level , y = predict(poly_reg,newdata = dataset)),
            color = 'blue')+
  ggtitle('truth or bluff (Polynomial Regression)')+
  xlab('Level')+
  ylab('Salary')


# predicting new result with Linear Regression
y_pred = predict(lin_reg,data.frame(Level = 6.5)) #here we have to create new dataset for single data that we want to predict by using data.dataframe we can add extra cell in our data to add specific value in it
# here data.frame(Level = 6.5)  actually created dataset with Level variable and 6.5 entry in Level column


# predicting new result with Polynomial Regression

y_pred = predict(poly_reg,data.frame(Level = 6.5,
                                    Level2 = 6.5^2,
                                    Level3 = 6.5^3,
                                    Level4 = 6.5^4))



# Visualising the Regression Model results (for higher resolution and smoother curve)
# install.packages('ggplot2')
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)  
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(poly_reg,
                                        newdata = data.frame(Level = x_grid,
                                                             Level2 = x_grid^2,
                                                             Level3 = x_grid^3,
                                                             Level4 = x_grid^4))),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Polynomial Regression)') +
  xlab('Level') +
  ylab('Salary')


