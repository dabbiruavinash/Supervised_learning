Use KNN model on Indian diabetes patients database and predict whether a new patient is diabetic (1) or not (0)

# prediction for diabetes using KNN algorithm
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# load the dataset from Notepad file
df = pd.read_csv('e:/test/diabetes.csv')
df

# values of 'Glucose', 'BloodPressure', etc cannot be accepted as zeros since they will effect the results
# let us replace such values with the mean of respective column

# take the list of columns where the 0s should be replaced
col_list = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']
for i in col_list:
     # replace 0s by Nan Values
     df[i] = df[i].replace(0, np.NaN)
     # calculate mean by skipping the rows having NaN values
     mean = int(df[i].mean(skipna = True))
     # replace NaN values by mean value
     df[i] = df[i],replace(np.NaN, mean)
df

# split the dataframe into features and target data
x = df.iloc[:, :8]
x

y = df.iloc[:, 8]
y

# split the data into train and test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0)

# lets us find k values and accuracy levels for each k
# append accuracy levels to scores list
k_range = range(1,20) # take k from 1 to 19
scores = []
for k in k_range:
      model = KNeighborsClassifier(n_neighbors=k)
      model.fit(x_train, y_train)
      accuracy = model.score(x_test, y_test)
      scores.append(accuracy)
      print('k = %d Accuracy = %.2f%%' % (k, accuracy*100))

# show the k value and scores in line plot
# we can see highest accuracy when k=14
import matplotlib.pyplot as plt
plt.plot(k_range, scores)
plt.xlabel("Value of k")
plt.ylabel("Accuracy")

# create KNN Classifier object with k = 14
model = KNeighborClassifier(n_neighbors=14, metric = 'euclidean')
model.fit(x_train, y_train)

# find accuracy
accuracy = model.score(x_test, y_test)
accuracy # 0.7922077922077922

# predict for the given data
model.predict([[1.189, 60, 23, 846, 30, 1, 0.398, 59]])
# output -> array([1]) -> diabetic

model.predict(([3,126,88,41,235,39.3, 0.704,27]])
# output -> array([0]) -> not a diabetic 