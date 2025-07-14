Given the money spent on Tv, radio and newspaper. find out which media has influenced sales more using Lasso Regression model.

# feature selection using Lasso Regression
df = pd.read_csv("E:/test/Advertising.csv")
df.head()

# see the names of columns
names = df.columns
names

# to delete the unamed column, first rename it as 'a' column then drop it as usual
df.rename({"Unnamed:0":"a"}, axis = 1, inplace = True)
df.drop(["a"], axis=1, inplace = True)

# drop sales column and take remaining columns as x
x = df.drop('sales','axis=1)
x

# take the names of 0,1 and 2nd columns
names = x.columns
names

# range of columns, The gives 0 to 3
rng = range(len(names))
rng

# apply Lasso regression on the data after the model is trained , find the coefficients 
from sklearn.linear_model import Lasso
ls = Lasso(alpha=0.01)
model = ls.fit(x,y)
cf = model.coef_
cf

# draw line plot between the column numbers and coefficients
import matplotlib.pyplot as plt
plt.plot(rng,cf)
plt.xticks(rng,name, rotation=60)
plt.ylabel("coefficients")
plt.show()

# find the accuracy of the model, gives 89.7%
ls.score(x,y) # 0.8972106012924924

' ' ' find the sales if the company is spending dollars 150 on Tv, 41 on radio and 60 on newspaper.
' ' '
newdata = [[150,41,60]]
sales = ls.predict(newdata)
sales # array([17.471243])
