Train the computer using simple linear regression model on the employee experience and salary data. Predict the salary of an employee having 11 years of experience

# Simple Linear Regression with train and test data
import pandas as pd
import matplotlib.pyplot as plt

#load the dataset from computer into dataset object
dataset = pd.read_csv("E:/test/Salary_Data.csv")

#retrieve only 0th column and take it as x
x = dataset.iloc[:, :-1].values

# retrieve only 1st column and take it as y
y = dataset.iloc[:, 1].values

# draw scatter plot to verify simple linear regression
# model can be used. Scatter plot show dots as straight line
plt.scatter(x,y)

# take 70% of data for training and 30% for testing
# random_state indicates the random seed used in selecting test rows
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test = train_test_split(x,y, test_size = 0.3, random_state = 0)

# train the computer with simple linear regression model
from sklearn.linear_model import LinearRegression

# create linear regression class object
reg = LinearRegression()

# train the model by passing train data to fit() method
reg.fit(x_train,y_train)

# test the model by passing test data and obtain predicted data
y_pred = reg.predict(x_test)

# find the r squared value by comparing test data
# (expected data) and predicated data. Accuracy is 97.4%
from sklearn.metrics import r2.score
r2_score(y_test, y_pred) #0.9740993407213511

# predict the salary of employee with experience 11 year
# this give 129740 dollars salary.
print(reg.predict(([[11]])) # [129740.26548933]

# using matplotlib, draw scatter plot and line plot
# select the below all statements and then execute at once
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, reg.predict(x_train), color = 'blue')
plt.title("experience vs salary")
plt.xlabel('Yrs of exp')
plt.ylabel('salary')
plt.show()

