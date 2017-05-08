# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('Salary_Data.csv')

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
#library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling    #we are not going to use feature scaling in simple linear regression bcz the library we are going to use will take care of that
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting Simple linear regression to Training set

regressor = lm(formula = Salary ~ YearsExperience,
               data = training_set) #after running this command we can see the summary of the regressor by using 'summary(regressor)' which includes some statistical information
 
# predicting the test set results

y_pred = predict(regressor,newdata = test_set)

# visualising Training set results 
  # here we are going to use ggplot2 package to plot the charts
ggplot() +
  geom_point(aes(x = training_set$YearsExperience,y = training_set$Salary),
             colour = 'red') + #here we plotted all training set points
  geom_line(aes(x = training_set$YearsExperience,y = predict(regressor,newdata = training_set)),colour = 'blue')+ # this will print regression line which showa the predicted values of taining set of Salaries
  ggtitle('Salary Vs Experience(Training set)') +
  xlab('Years of experience')+
  ylab('Salaries')
  # our model was built on this training set which gave satisfying results though now lets see how it works on test set 

# visualising Test set results
# here we are going to use ggplot2 package to plot the charts
ggplot() +
  geom_point(aes(x = test_set$YearsExperience,y = test_set$Salary),
             colour = 'red') + #here we plotted all training set points
  geom_line(aes(x = training_set$YearsExperience,y = predict(regressor,newdata = training_set)),colour = 'blue')+ # this will print regression line which showa the predicted values of taining set of Salaries
  ggtitle('Salary Vs Experience(Test set)') +
  xlab('Years of experience')+
  ylab('Salaries')