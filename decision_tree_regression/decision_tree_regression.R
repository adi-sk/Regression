# Decision Tree Regression 

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Splitting the dataset into the Training set and Test set
# # install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Salary, SplitRatio = 2/3)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting the Decisioin Tree Regression Model to the dataset
# Create your regressor here
# install.packages('rpart')
# library(rpart)
regressor = rpart(formula = Salary ~ . ,
                  data = dataset,
                  control = rpart.control(minsplit = 2))

# Predicting a new result
y_pred = predict(regressor, data.frame(Level = 6.5)) # this may not predict the right result So to conclude Decision Tree regression model is not the interesting model in 1D but it is more powerful model in more dimensions

# Visualising the Decisioin Tree Regression Model results NOTE : this does not give the right graph see python explaination solution for this is to visualize the graph in high resolution 
#install.packages('ggplot2')
#library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(regressor, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Decisioin Tree Regression Model)') +
  xlab('Level') +
  ylab('Salary')

# Visualising the Decisioin Tree Regression Model results (for higher resolution and smoother curve)
# install.packages('ggplot2')
#library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Decisioin Tree Regression Model)') +
  xlab('Level') +
  ylab('Salary')