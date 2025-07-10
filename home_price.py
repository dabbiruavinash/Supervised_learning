We are going home prices in monroe township, NJ (USA). We should predict the prices for the following homes:
a. 3000 sqft area, 3 bed rooms, 40 years old
b. 2500 sqft area, 4 bed rooms, 5 years old

# multiple linear regression model - predicitng house prices
import pandas as pd

# load the dataset into dataframe
df = pd.read_csv("E:/test/homeprices.csv")

# find out any missing value in the dataset
# bedrooms has missing value
df.isnull().sum()

# full the missing data (NaN) with median of bedrooms
import math
med = math.floor(df.bedrooms.median())
med #4

# fill the missing data (NaN columns) with this median value
df.bedrooms = df.bedrooms.fillna(med)
df

# represent the relations between independent and dependent vars
# area, bedrooms and age are independent vars and price is dependent
import seaborn as sns
sns.lmplot(x = 'area', y='price', data=df)
sns.lmplot(x= 'bedrooms', y='price', data=df)
sns.lmplot(x='age', y='price', data=df)

#create linear regression model with mutliple variables
# take the independent vars first and take dependent var next.
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(df[['area','bedrooms','age']], df['price'])

# print coefficients i.e. m1, m2, m3 values
reg.coef_ #142.895644 m -48591.66405516, -8529.30115951

# interceot
reg.interccept_ # 485561.8928233806

# predict the price of 3000sqft area, 3 bedrooms, 40 years old house
reg.predict([[3000,3,40]]) # 427301

# predict the price of 2500 sqft area, 4 bed rooms, 5 years old house
reg.predict([[2500,4,5]]) # 605787
