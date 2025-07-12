carprices.csv contains car sales prices for 3 different models. First , plot data points on a regression plot to see if Linear Regression plot to see if linear regression model can be applied, Then build a model that can answer the following questions:
1. Predict price of a Mercedez Benz that is 4years old with mileage 45000.
2. Predict price of a BMW X5 that is 7years old with mileage 86000
3. What is the score (accuracy) of your model.

# one hot encoding on care dataset
import pandas as pd
df = pd_csv("E:/test/carprices.csv")
df

# check if linear regression can be used on this data
# They are showing approximate straight lines
import seaborn as sns
sns.regplot(data = df, x = 'Mileage', y = 'Sell Price($)')
sns.regplot(data = df, x = 'Age(yrs)', y = 'Sell Price($)')

# to use one hot encoding , first we should use Label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# fit and transform the data frame using le on Car Model column
# see the new data frame where car model will be 1,0,2
# here , 1 = BMW X5, 0 = Audi A5, 2 = Mercedez Benz c class
df['Car Model'] = le.fit_transform(df['CarModel'])
df['Car Model']

# take independent variable or features
x = df[['car model', 'Mileage', 'Age(yrs)']]
x

# retrieve target or dependent variable
y = df['Sell Price{$)']
y

# now apply OneHotEncoding on car model column
from sklearn.preprocessing import OneHotEncoding

ohe - OneHotEncoder(handle_unknown = 'ignore')
x1 = ohe.fit_transform(df[['Car Model']])

# x1 is a sparse matrix. convert it into array and dataframe
# here , 010 = BMW X5, 100 = Audi A5, 001 = Mercedez Benz C class
x1 = pd.DataFrame(xq.toarray())
x1

# to avoid dummy variable trap, drop 0th column
x1 = x1.iloc[:, 1:]
x1

# add x1 columns to x
x = pd.concat([x,x1], axis = "columns")
x

# remove car model column as it is available in encoded form (the last two columns)
x.drop('Car Model', axis = 1, inplace = True)
x

# let us create linear regression model
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# train the model
model.fit(x,y)

# predict price of a mercedez benz that is 4years old
# with mileage 45000. observe x data , 10 = BMW X5 , 00 = Audi A5,
# 01 = Mercedez Benz C class
model.predict([[45000,4,0,1]]) # array([36991.31721061])

# predict price of a BMW X5, that is 7yers old with mileage 86000
model.predict([[86000,7,1,0]]) # array([11080.74313219])

# know the score (accuracy) of your model. It is 94.1%
model.score(x,y) # 0.9417050937281083