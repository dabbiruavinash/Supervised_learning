Find out the per captia income of canada during the years 2020 and 2021 based on the per capita income data provided using simple linear regression model

# training the machine with simple linear regression model
import matplotlib.pyplot as plt
import pandas as pd

# read data from "sheet1" of excel file
dataset = pd.read_excel("E:/test/canada_per_capita_income_income.xlsx", "Sheet1")

x = dataset.iloc[:, 0:1].values
x

y = dataset.iloc[:, -1].values
y

#check the distribution of data is linear or not
plt.scatter(x,y, color = 'red')

# take 70% of data for training and 30% for testing
# random_state indicates the random seed used in splitting the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 5)

# train the computer using simple linear regression model
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train, y_train)

# make prediction based on test data
y_pred = reg.predict(x_test)

# find the r squared value by comparing test data and predicted data 
from sklearn.metrics import r2_score
r2_score(y_test, y_pred) # 0.843326110551844

# another way to know the score of a linear regression model
reg.score(x_test, y_test) # 0.843326110551844

# predict the per capita income during the years 2020 and 2021
# output: array*[41819.49650873, 42681.02869595])
# this means 41819 $ in 2020 and 42681$ in 2021 years.
reg.predict(([2020], [2021]])

# draw scatter plot and regression line
# select below block and run at once
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, reg.predict(x_train), color = 'blue')
plt.title("PER CAPITA INCOME OF CANADA")
plt.xlabel('Year')
plt.ylabel('Per Capita Income')
plt.show()
