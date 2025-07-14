Using Lasso Regression on Boston house price, select the best feature that mostly influences the output(price of the house)

# feature selection using lasso Regression
import pandas as pd
boston = pd.read_csv("E:/test/boston_house.csv")
baston.head()

# see the names of columns
names = boston.columns
names

# to delete the unamed column, first rename it as 'a' column then drop it as usual
boston.rename({"unamed: 0":"a"}, axis="columns",inplace=True)
boston.drop(["a"], axis=1, inplace=True)

# drop MEDV column and take remaining columns as x
x = boston.drop('MEDV', axis = 'columns')
x

# take MEDV column as y
y = boston['MEDV'].values
y

# take the names of 0 to 13th columns
names = x.columns
names

# range of columns. The gives 0 to 13
rng = range(len(names))
rng

# apply Lasso regression on the data after the model is trained, find the coefficients
from sklearn.linear_model import Lasso
ls = Lasso(alpha=0.1)
data = ls.fit(x,y)
cf = data.coef_
cf

# draw line plot between the column numbers and coefficients 
import matplotlib.pyplot as plt
plt.plot(rng, cf)
plt.xticks(rng, names, rotations=60)
plt.ylabel("coefficients")
plt.show()

# find the accuracy of the model gives 72.6%
data.score(x,y) # 0.7269834862602695
' ' '
find the price (MEDV) of the house when following data is given:
CRIM = 0.02731 , zn = 0, INDUS = 7.07, CHAS = 0, NOX = 0.469, RM = 6.421, AGE = 78.9, DIS = 4.9671, RAD = 2, TAX = 242, PTRATIO = 17.8, BLACK = 396.9, LSTAT = 9.14
' ' '

newdata = [[0.02731, 0, 7.07, 0.469, 6.421, 78.9, 4.9671, 2, 2.242, 17.8, 396.9, 9.14]]
price = data.predict(newdata)
price # array([24.71246809])
