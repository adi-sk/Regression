# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('50_Startups_MultipleLinearRegression.csv')

#Encoding Categorical Data
dataset$State = factor(dataset$State,
                       levels = c('New York','California','Florida'),
                       labels = c(1,2,3))


# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
#library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling  # feature scaling is not required here library will take care of that
# training_set = scale(training_set)
# test_set = scale(test_set)

#Fitting Multiple Linear Regression to Training set

#regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State)# R replaces ' ' with '.'
                          # OR we can use
regressor = lm(formula = Profit ~ ., # '.' includes all the independent variables
               data = training_set)  # the best part of the R regressor is it doesn't fall in dummy variable trap it automatically removes one dummy variable while forming model
summary(regressor) # if we see the summary we can find that only siginificant variable here is 'R.D.Spend' it is the only variable which has p value < 0.05

# that means we can replace our regression formula with Profit ~ R.D.Spend


# predicting Test set Results

y_pred = predict(regressor,newdata = test_set) # even if we use formula Profit ~ R.D.Spend we will get the same results in y_pred bcz R.D.Spend is the only Significant variable here