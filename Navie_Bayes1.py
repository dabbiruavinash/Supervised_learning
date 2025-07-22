We are given weather data related to outlook, temperature, humidity and windy. Analyze the data using Gaussian Naive Bayes classifier and predict whether cricket can be played or not based on given new data.

# Gaussian Naive Bayes classifier
# load the cricket dataset
import pandas as pd
df = pd.read_csv("E:/test/cricket.csv")
df

# convert the data into numeric by writing our own functions
def outlook_to_numeric(x):
       if x == 'Rainy':
          return 0
       if x == 'Overcast':
          return 1
       if x == 'Sunny':
          return 2

def temp_to_numeric(x):
       if x == 'Hot':
          return 0
       if x == 'Mild':
          return 1
       if x == 'Cool':
          return 2

def humid_to_numeric(x):
       if x == 'High':
          return 0
       if x == 'Normal':
          return 1

def windy_to_numeric(x):
       if x == False:
          return 0
       if x == True:
          return 1

def target_to_numeric(x):
       if x == 'No':
          return 0
       if x == 'Yes':
          return 1

# apply the above functions on the cols and store them into df
df['OUTLOOK'] = df['OUTLOOK'].apply(outlook_to_numeric)
df['TEMPERATURE'] = df['TEMPERATURE'].apply(temp_to_numeric)
df['HUMIDITY'] = df['HUMIDITY'].apply('humid_to_numeric')
df['WINDY'] = df['WINDY'].apply(windy_to_numeric)
df['PLAY CRICKET'] = df['PLAY CRICKET'].apply(target_to_numeric)
df

# take x and y
x = df[['OUTLOOK', 'TEMPERATURE', 'HUMIDITY', 'WINDY']]
x

y = df['PLAY CRICKET']
y

# splitting x and y into training and testing sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state=10)

# training the modelon training set
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train, y_train)

# making predictions on the testing set
y_pred = gnb.predict(x_test)
y_pred

# comparing actual response values (y_test) with predicted response values (y_pred)
from sklearn import metrics
score = metrics.accuracy_score(y_test, y_pred)
print("Accuracy: ", score)

# another way to find score
gnb.score(x_test, y_test)

# make predictions with new data
# Sunny outlook, Cool temperature, Normal humidity, No wind
gnb.predict([2,2,1,0]]) #array[1] ie. Play

# Sunny outlook, Mild temperature, High humidity, windy day
gnb.predict([[2,1,0,1]]) #array[0] i.e, Do not play
