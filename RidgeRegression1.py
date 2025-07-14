Ridge Regression to predict height depending on weight 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge

# load train data into x and test data into y
x = pd.read_csv("E:/test/train.csv")
x

y = pd.read_csv("E:/test/test.csv")
y

# take the data into x and y variable
w_train = x.iloc[:,0:1].values
h_train = x.iloc[:, 1].values
w_test = y.iloc[:,0:1].values
h_test = y.iloc[:,1].values

# draw scatter plot for train and test data in the same graph
plt.scatter(w_train,h_train,color='blue', marker='o')
plt.scatter(w_test, h_test, color='red', marker='d')
plt.show()

# using simple linear regression to fit train data
lr = LinearRegression()
lr.fit(w_train, h_train)

# find the score of the simple Linear Regression on train data
# this gives 0.9554154968329801 that is around 95.5% accuracy
lr.score(w_train, h_train)

# draw scatter plot with original data and predicted line
plt.scatter(w_train, h_train, color='blue')
plt.plot(w_train, lr.predict(w_train), color='orange')

# using Ridge Regression to fit a line for test data take alpha values as 0.001, 0.01, 0.05, 0.5, 1, 2, etc.
rr = Ridge(alpha=0.001)
rr.fit(w_test, h_test)

# find the score of the Ridge Regression on test data this gives 0.8901155974442 that is around 89% accuracy
rr.score(w_test, h_test)

# draw scatter plot with original data and predicted line
plt.scatter(w_test,h_test,color='red')
plt.plot(w_test, rr.predict(w_test), color = 'green')
plt.show()

# predict height of a person having 70.3 kg weight this gives [6.27334337] that is approximately 6.27 feet
rr.predict([[70.3]]) # 6.27ft

