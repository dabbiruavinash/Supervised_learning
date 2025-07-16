Apply Decision Tree classifier machine learning model to take a decision whether to play cricket or not under given conditions.

# deciding to play cricket or not using a decision tree
import pandas as pd

# load the dataset
df = df.read_csv('e:/test/cricket1.csv')
df

# let us convert the column data into numeric
# this is done with LabelEncoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# apply label encoder on all columns.
# the following conversion takes place
' ' ' Outlook ---> 0 overcast, 1 Rainy, 2 sunny
      Temperature ---> 0 cool, 1 Hot, 2 mild
      Humidity ---> 0 High, 1 Normal
      Windy ---> 0 False, 1 True
      Play Cricket ---> 0 No, 1 Yes
' ' '

df['Outlook'] = le.fit_transform(df['Outlook'])
df['Temp_n'] = le.fit_transform(df['Temperature'])
df['Humidity_n'] = le.fit_transform(df['windy'])
df['Play_n'] = le.fit_transform(df['Play Cricket'])
df

# delete cols with labels (or strings)
df = df.drop(['Outlooks', 'Temperature', 'Humidity', 'Windy', 'Play', 'Cricket'], axis = 'Columns')
df

# divide the data into x and y
x = df.drop(['Play_n'], axis = 'columns')
y = df['Play_n']

# create the DecisionTreeClassifier model
from sklearn.tree import DecisionTreeClassifier

# default crterion = 'gini'. we can use criterion = 'entropy' also.
model.fit(x,y)

# predict whether to play cricket or not for following data:
# today = (Outlook=Sunny, Temperature=Not, Humidity=High, Windy=FALSE)
model.predict([2,1,0,0]) # array([0]) - - -> No