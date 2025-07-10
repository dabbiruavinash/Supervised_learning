Predicitng the price of a house in new york depending on its area. Given the house area. find out the prices whose area are: 3300 sqft and 5000 sqft

# Simple Linear Regression
# predicting the house prices depending on area

import pandas as pd
import seaborn as sns
import sklearn.linear_model import LinearRegression

# Load the data into dataframe
df = pd.read_csv("E:/test/homeprice.csv")
df.show()

# plot a scatter plot
sns.scatterplot(data=df, x='area', y='price')

# once we see the scatter plot, we can understand that the distribution is linear and can use Linear Regression Model
reg = LinearRegression()
reg.fit(df[['area']], df.price)

# predict the price of 3300 sft house
reg.predict([[3300]])

# find the coefficient. this is slope m
reg.coef_

# find the intercept. this is b
reg.intercept_

# if we substitute m and b values in y = mx+b
# we get the predicted value above

y = 135.78767123 * 3300 + 180616.43835616432
y # display 628715.7534151643

# next predict the price of 5000 sft house
reg.predict([[5000]]) #859554.79452055

# find accuracy level of the model by finding r squared values
# gives 95.8% accuracy
y_original = df.price
y_predicted = reg.predict(df[['area']])

R_square = r2_score(y_original, y_predicted)
print('r squared value', R_square)

# display the scatter plot with a regression line
sns.lmplot(data = df, x = 'area', y = 'price')
