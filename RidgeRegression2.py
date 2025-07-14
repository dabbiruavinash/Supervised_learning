Use boston_house.csv and take only 'RM' and 'Price' of the house. Divide the data as training and testing data. Fit the line using Ridge regression and find the price of a house if it contains 6.5 rooms.

# Ridge Regression to predict price of a house depending on number of rooms
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge

# load the dataset into df object
df = pd.read_csv("E:/test/boston_houses.csv")
df

# take RM (5th column) into x and convert into 2D array
x = df.iloc[:, 5:6].values

# take price (last column) into y and convert into 1D array
# iloc[:,-1] represents all rows in last column
y = df.iloc[:, -1].values

# split the data into train and test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_sized=0.2, random_state=1)

# draw scatter plot for train and test data in the same graph
plt.scatter(x_train, y_train, color = 'blue', marker='^')
plt.scatter(x_test, y_test,color='red', marker='+')
plt.show()

# using simple Linear Regression fit a line for the train data
lr = LinearRegression()
lr.fit(x_train, y_train)
lr.score(x_train, y_train)

# display the train data with regression line
plt.scatter(x_train, y_train, color='blue', marker = '^')
plt.scatter(x_train, lr.predict(x_train), color = 'red')

# using Ridge regression to fit a line for test data take alpha values as : 0.001,0.01,0.5,1,2, etc
rr = Ridge(alpha = 0.01)
rr.fit(x_test, y_test)
rr.score(x_test, y_test)

# display the test data with regression line
plt.scatter(x_test, y_test, color = 'red', marker='+')
plt.plot(x_test, rr.predict(x_test), color='green')
plt.show()

# predict price of a house 6.5 rooms
rr.predict([[6.5]]) # 24.1 k dollars
