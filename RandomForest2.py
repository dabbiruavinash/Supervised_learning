Train teh Random Forest Classifer Machine Learning model on Employee experience and salary dataset and predict the salary of an employee when his experience is given

# load the dataset
import pandas as pd
df = pd.read_csv("E:/test/salary_Experience.csv")
df

# take 1st column, i.e, YearsExperience as x
# This should be in 2D array format
# take salary as y.This should be in 1D array format

x = df.iloc[:, 0:1]
y = df['Salary'].values

# import Random Forest Regression Class
from sklearn.ensemble import RandomForestRegressor

# create Random Forest Regressor class object
regressor = RandomForestRegressor(n_estimators = 100, random_state=0)

# fit the regressor with x and y data
regressor.fit(x,y)

# what is the score with 100 trees?
regressor.score(x,y) # 0.992366946643155

# make prediction
# find out the salary of an employee with 4.2 years experience
regressor.predict([[4.2]]) # array([56932.77516667])

# find out the salary of an employee with 11 years experience
regressor.predict(([7]]) # array([98687.5])

# visualizing the results
# scatter plot shows original data points. Line plot shows how random forest regressor is predicting accordingly.
# select the below code and then execute as a single unit
import matplotlib.pyplot as plt
plt.scatter(x,y,color='blue')
plt.plot(x, regressor.predict(x), color = 'red')
plt.title('Random Forest Regression')
plt.xlabel('Experience level')
plt.ylabel('Salary')
plt.show()