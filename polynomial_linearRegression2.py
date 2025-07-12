Take Salary_Experience.csv dataset and do analysis with Polynomial Linear Regression for the degree 5. FInd the following
a. draw the scatter plot with regression line
b. find coefficients 
c. intercept
d. accuracy of your model and 
e. judge the salary of an employee having 5.5 years of experience

# polynomial regression on salary and experience data
import matplotlib.pyplot as plt
import pandas as pd

# load the dataset into dataframe object
df = pd.read_csv('E:/test/Salary_Experience.csv')
df

# retrieve 0th column as x
x = df.iloc[:, 0:1].values
x

# retrieve lst column as y
y = df.ilc[:,-1].values

# draw scatter plot to know data points are distributed they are in wave form
# plt.scatter(df['YearsExperience'], df['Salary'])
plt.scatter(x,y)
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Salary v/s Years of Experience')
plt.show()

# create the polynomial Features with degree 5
from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=5)

# convert x using the polynomial features of degree 5
x1 = pf.fit_transform(x)

# create the polynomial Regression Model
from sklearn.linear_model import LinearRegression
lr = LinearRegression

# train the model on x1 and y
lr.fit(x1,y)

# calculate the accuracy with degree = 5
lr.score(x1, y) # 0.965446487572477

# draw data points along with regression line
plt.scatter(x,y)
plt.plot(x, lr.predict(x1))

# find the coefficient
# since degree is 5, we will have 5 coefficients
print('Coefficients: ' lr.coef_)

# find intercept
# we will have only 1 intercept
print('Intercept :', lr.intercept_)

# judge the salary of employee with 5.5years experience
# input should be in 2D and output will be in 1D array form
input = [[5,5]]
input1 = pf.fit_transform(input)
result = lr.predict(input1)
print("Salary for 5.5yrs experience = ", result) # [79328.86462428]
