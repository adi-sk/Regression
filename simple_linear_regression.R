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


