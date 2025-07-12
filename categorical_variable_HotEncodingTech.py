A python program to understand how to encode categorical variables using one hot encoding technique.

# one hot encoding using sklearn OneHotEncoder
import pandas as pd
df = pd.read_csv("E:/test/homeprices.csv")
df

# to use one hot encoding, first we should use label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# fit and transform the data frame using le on town column
# see the new data frame where town will be 0,2 or 1
df.town = le.fit_transform(df.town)
df

# retreive training data
x = df[['town','area']]
x

# retrieve target data
y = df.price
y

# apply one hot encoding on town column
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(handle_unknown='ignore')

x1 = ohe.fit_transform(df[['town']])
x1 = pd.DataFrame(x1.toarray())
x1

# to avoid dummy variable trap, drop 0th column
x1 = x1.iloc[:,1:]

# add these columns to x
x = pd.concat([x,x1], axis = "columns")
x

# remove town as it is already encoded
x.drop('town', axis = 1, inplae=True)

# lets us create linear regression model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x,y) # train the model

# predict the price of house with 2800 sft area
# location at robinsvile
model.predict([[2800, 1,0]]) # array([590775.63964739])

# predict the price of house at monroe township
model.predict([[3400,0,0]]) # array([641227.69296925])

# final accuracy of the model
model.score(x,y) # .9573929037221871