Import required Libraries:

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import math

sns.set_theme(color_codes=True)

Load Dataset:

url = "https://raw.githubusercontent.com/tajamulkhann/Machine-Learning-Projects/main/Supervised%20Learning%20Projects/Airline%20Satisfaction%20Prediction%20using%20Classification%20Algorithms/ABNB.csv"

df = pd.read_csv(url)
df.head()

Convert data column to datetime format:

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplacce=True)
df.head()

Exploratory Data Analysis:

Visualizing stock price trends:

plt.figure(figsize = (12,6))
plt.plot(df.index, df['Close'], label = 'Closing Price', color = 'b')
plt.xlabel("Date")
plt.ylabel("Stock Price (USD)")
plt.title("Airbnb Stock Closing Price Over Time")
plt.legend()
plt.xtricks(rotation=90)
plt.show()

Feature Engineering: Extracting Date Features

def create_features_datetime(df):
      df['Year'] = df.index.year
      df['month'] = df.index.month
      df['DayOfWeek'] = df.index.dayofweek
      return df

df = create_features_datetime(df)
df.head()

Machine Learning Model for stock price prediction

splitting data into train and test sets:

x = df.drop(columns = ['Close'])
y = df['Close']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

Model 1: Decision Tree Regressor

dtree = DecisionTreeRegressor(random_state=0)
dfree.fit(x_train, y_train)

y_pred_dt = dtree.predict(x_test)

mae_dt = metrics.mean_absolute_error(y_test, y_pred_dt)
mse_dt = metrics.mean_squared_error(y_test, y_pred_dt)
r2_dt = metrics.r2_score(y_test, y_pred_dt)
rmse_dt = math.sqrt(mse_dt)

print(f'Decision Tree - MAE: {mae_dt}, MSE: {mse_dt}, R2: {r2_dt}, RMSE: {rmse_dt}')

Model 2: Random Forest Regressor

rf = RandomForestRegressor(random_state=0)
rf.fit(x_train, y_train)

y_pred_rf = rf.predict(x_test)

mae_rf = metrics.mean_absolute_error(y_test, y_pred_rf)
mse_rf = metrics.mean_squared_error(y_test, y_pred_rf)
r2_rf = metrics.r2_score(y_test, y_pred_rf)
rmse_rf = math.sqrt(mse_rf)

print(f'Random Forest - MAE: {mae_rf}, MSE: {mse_rf}, R2: {r2_rf}, RMSE: {rmse_rf}')

Visualizing Predictions

Adding Predicitions to the dataset:

df['Predicted_Close'] = rf.predict(X)

Plot Acutal vs Predicted Stock Prices:

plt.figure(figsize = (12,6))
plt.plot(df.index, df['Close'], label = 'Actual Close Price', color = 'blue')
plt.plot(df.index, df['Predicted_Close'], label = 'Predicted Close Price', color = 'red', linestyle='dashed')
plt.xlabel("Date")
plt.ylabel("Stock Price (USD)")
plt.title("Airbnb Stock Price Prediction using Random Forest")
plt.legend()
plt.xticks(rotation=90)
plt.show()



