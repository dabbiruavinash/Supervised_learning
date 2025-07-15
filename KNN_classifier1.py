Given the data of breat cancer patients. FInd out wheather the cancer is bengin or malignant with the help of K Neighbors Machine Learning Model.

# prediction for breat cancer using KNN classifier 
import pandas as pd

# load the data from data file into the data frame
df = pd.read_csv('e:/test/breast-cancer-wisconsin.data')
df.head()

# display the column names
df.columns

# since the column names having spaces , remove them
# convert column into strings and then replace the space with empty string
df.columns = df.columns.str.replace(' ','')
df.columns

# we can find? mark in the bare_nulei column.
# find out such rows - there are 16 such rows
df[df['bare_nulei] = '?']

# copy those rows into df where '?' is not found
df = df[df['bare_nulei'] ! = '?']

# remove uselsee data. Here it is id col
# 1 indicates drop label from columns.
# 0 indicates drop label from index
df.drop(['id'], axis = 1, inplace = True) # axis = 'columns'
df

# take 0 to 8th cols in x
x = df.iloc[:, :9]
x

# take 9th column, i.e, class column in y
y = df.iloc[:, 9] # y can be 2 or 4
y

# split the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

# k value is square root of number of rows in test data
import math
k = math.sqrt(len(y_test))
k # 11.704699910719626

# see k value is even, convert it to odd
if k % 2 == 0: k+=1
k = int(k)
k #11

# import KNeighborsClassifier class
from sklearn.neighbors import KNeighborsClassifier

# create the model with k value obtained above
model = KNeighborsClassifier(n_neighors=k)

# find accuracy
accuracy = model.score(x_test, y_test)
accuracy # 0.9560243795620438

# let us find out where highest accuracy can be achieved
# let us find k values and accuracy levels for each k
k_range = range(1,16)
scores = []
for k in k_range:
      model = KNeighborsClassifier(n_neighbors=k)
      model.fit(x_train, y_train)
      accuracy = model.score(x_test, y_test)
      scores.append(accuracy)
      print('k = %d Accuracy = %.2f%%' % (k, accuracy*100))

# show the k values and scire in line plot
# we can see highest accuracy when k = 1,3,5,7
import matplot.pyplot as plt
plt.plot(k_range, scores)
plt.xlabel("value of k")
plt.ylabel("Accuracy")

# Take k=3 for achieveing highest accuracy.
model = KNeighborsClassification(n_neighbors=3)
model.fit(x_train, y_train)

# find accuracy
accuracy = model.score(x_test, y_test)
accuracy # 0.9708029197080292

# with the above highest accuracy, we can make predictions
# predict for the given data
model.predict([[4,2,1,1,1,2,3,2,1]])
# output -> array([2]) -> bengin cancer

# predict for two patients. 2-> benign, 4-> malignant
model.predict([[4,2,1,1,1,2,3,2,1], [8,10,10,8,7,10,9,7,1]])
# output -> array([2,4]) -> 1st set shows benign cancer and the 2nd set of data shows malignant cancer