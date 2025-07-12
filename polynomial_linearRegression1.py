Fit the simple linear regression and polynomial linear regression models to salary_position.csv data. Find which one is more accurately fitting to the given data. Also predict the salaries of level 10 and 11 employees.

# polynomial linear regression
import pandas as pd
import matplotlib.pyplot as plt

# read the file content into dataframe
df = pd.read_csv("E:/test/Salary_Position.csv")

# retrieve level col, ie 1st column as 2D array
x = df.iloc[:, 1:2].values

# retrieve Salary col, ie 2nd column as 1D array
y = df.iloc[:,2].values
y

# let us scatter plot the distribution of the data points
plt.scatter(x,y,color='red')
plt.show()

# create linear regression model
from sklearn.linear_model import LinearRegression
slr = LinearRegression()
slr.fit(x,y) # fitting means training

# predict y values based on x using the model
y_pred = slr.predict(x)

# now plot the simple linear regression line on the data points
plt.scatter(x,y,color='red')
plt.plot(x, y_pred, color = 'blue')
plt.show()

# create polynomial linear regression model upto the term x^2
from sklearn.preprocessing import polynomialFeatures
pf = PolynomialFeatures(degree=2)

# convert x into polynomial regression with degree 2
x1 = pf.fit_transform(x)

# apply polynomial linear regression on x1
plr.fit(x,y)

# predict y values based on x1
y_predict = plr.predict(x1)

# plot the polynomial linear regressioin - this fits best
plt.scatter(x,y, color = 'red')
plt.plot(x, y_pred, color='blue')
plt.show()

# improving the polynomial linear regression model
# increase degree value from 1and 8 and see the results again

for i in range(1,9):
      pf = polynomialFeatures(degree=i)
      x1 = pf.fit_transform(x)
      plr.fit(x1,y)
      plt.scatter(x,y,color='red')
      plt.plot(x, plr.predict(x1), color = 'blue')
      plt.show()
      print('----------------------------------------------')

# use degree 7 for polynomial Linear Regression model
pf = PolynomialFeatures(degree=7)

# convert x into polynomial regression with degree 7
x1 = pf.fit_transform(x)

# apply polynomial linear regression on x1
plr = LinearRegression()
plr.fit(x1,y)

# observe the r squared value. best fit ig it is closer to 1
from sklearn.metrics import r2_score
r2_score(y,slr.predict(x)) # 0.5537636591968075
r2_score(y,plr.predict(x1)) # 0.9984370801255893

# accuracy can also be obtained using score() functions
slr.score(x,y)
plr.score(x1,y)

# predict salary for level 10 and 11 employees
# output: array([569267.44462639, 903162.29485398])
inputs = [[10],[11]]
inputs1 = pf.fit_transform(inputs)
plr.predict(inputs1)