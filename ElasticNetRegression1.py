select the most appropiate feature of Boston housing dataset using Elastic Net Regression model.

# feature selection using elastic net regression
import pandas as pd
boston = pd.read_csv("E:/test/boston_houses.csv")
boston.head()

# delete the price column as it is dependent and take it as y
x = boston.drop('MEDV', axis = 1)
x

y = boston['MEDV'].values
y

# take the name of columns - gives range(0,13)
rng = range(len(names))
rng

# create an object to ElasticNet Class
from sklearn.linear_model import ElasticNet
es = ElasticNet(l1_ratio = 0.5)

# train the model
model = es.fit(x,y)

# find the coefficient of term in elastic net model equation the term with highest coefficient will exert with influence on the house value
es_coef = model.coef_
es_coef

# find the position of maximum value in the coefficients
import numpy as np
n = np.argmax(es_coef)
n # gives 5

# find the column name at nth position in x.
print('The most influencing column =', names[n])

# draw line plot between range and coefficients
# we can see highest peak at 'RM' column
plt.plot(rng, es_coef)
plt.xticks(rng, names, rotation=60)
plt.ylabel("Coefficients")
plt.show()


