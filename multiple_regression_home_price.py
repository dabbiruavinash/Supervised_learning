create multiple linear regression model for the house price data set. Divide the dataset into train and test data while giving it to the model and predict the prices of the house with the following specifications.

1. Predict the house price with 780 sqft area, 3 bedrooms and 1 bath room.
2. predict the house prices for two house with 1500 sqft, 3 bed rooms and 2 bath rooms and another one with 2000 sqft , 4 bed rooms and 4 bath rooms.

# multiple linear regression - predicting house prices
import pandas as pd

# load the dataseet into dataframe
df = pd.read_excel("E:/test/call-03homes.xls", "Sheet1")
df

# find out any missing values in the dataset there are no missing value in any column
df.isnull().sum()

# find out outliers by drawing box plots - there are outliers in price 
import seaborn as sns
sns.lmplot(x ='SqFt', y='Price', data=df)
sns.boxplot(x='BedRooms', y='Price', data=df)
sns.boxplot(x='Baths', y='Price', data=df)

#delete the rows with outliers using iqr method
# calculate q3 (third quartile).
q3 = df['price'].quantile(0.75)
q3

# calculate q1 (first quartile)
q1 = df['Price'].quantile(0.25)

# find iqr value. this gives 80000.0
iqr = q3 - q1
iqr

# calculate upper and lower limits from iqr
ul = q3 + (1.5 * iqr)
ll = q1 - (1.5 * iqr)
print(ul, ll) # 304900.0  -15100.0

# upper bound
import numpy as np
upper = np.where(df['Price'] >= ul)

# Lower bound
lower = np.where(df['Price'] <= ll)

# delete the rows above upper and below lower values
df.drop(upper[0], inplace = True)
df.drop(lower[0], inplace = True)

# retrieve only 2nd, 3rd and 4th columns and take then as x
x = df.iloc[:, 2:5].values

# retrieve lst column and take it as y
y = df.iloc[:, 1].values
y

# take 80% of data for training and 20% for testing random_state indicates the random seed used in selecting test rows
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=10)

# train the computer with simple linear regression model
from sklearn.linear_model import LinearRegression

# create linearRegression class object
reg = LinearRegression()

# train the model by passing train data to fit() method
reg.fit(x_train, y_train)

# test the model by passing test data and obtain predicted data
y_pred = reg.predict(x_test)

# find the r squared value by comparing test data (expected data) and predicted data. Accuracy is 97.4%
from sklearn.metrics import r2_score
r2_score(y_test, y_pred) # 0.829686049412441

# another way to find the score
reg.score(x_test, y_test) #  0.829686049412441

# predict the price of a house with 780 sqft, 3 bedrooms and 1 bathroom
# this gives 56120 dollars.
print(reg.predict([[780,3,1]]) # 56120.32684253

# predict the price of house with 1522 sqft, 3 bedrooms and 2 bathrooms and 2000 sqft, 4 bedrooms and 4 bathrooms
#output: 128866 dollars and 205294 dollars.
print(reg.predict([[1500, 3, 2], [2000, 4, 4]]))