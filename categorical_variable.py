A python program to understand how to encode categorical variables using dummy variable method

# dummy variable method - using pandas get_dummies()

import pandas as pd
df = pd.read_csv("E:/test/homeprices.csv")
df

# create dummy variable
dummies = pd.get_dummies(df.town)
dummies

# to avoid dummy variable trap, remove 'west windsor'
dummies = dummies.drop(['west windsor'], axis = 'columns')
dummies

# add these dummies to original df. add columns of both
merged = pd.concat([df,dummies], axis = 'columns')
merged

# we do not require 'town' variable as it is replaced by dummy vars.
# hence drop town
final = merged.drop(['town'], axis = 'columns')

# we have to deleted price column as it is the target column to be predicted
x = final.drop(['price'], axis = 'columns')
x

y = final['price']
y

# even though we do not drop the dummy variable
# linear regression model will work correctly
# the reason is it will internally drop a column

# let us create linear regression model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x,y) # train the model

# predict the price of house with 2800 sft area located at robinsville
# parameters: 1st: area, 2nd: monroe township, 3rd: robinsville
model.predict([[2800,0,1]]) # array([590775.6364739])

# predict the price of house with 3400 sft at west windsor
model.predict([[3400,0,0]]) # array([681241.66845839])

# find the accuracy of our model. Giving 95.7% accuracy
model.score(x,y) #0.9573929037221873

