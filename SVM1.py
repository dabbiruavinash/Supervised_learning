Classify the iris flowers data set using SVM and find out the flower type depending on the given input data

# iris flower classification using svm
from sklearn.dataset import load_iris
iris = load_iris()

# display info about the dataset
dir(iris) # display the attribute names of the dataset
iris.data # display data
iris.target # 0 - satosa, 1 - versicolor, 2 - virginica
iris.target_names # flower names
iris.feature_name # feature (column) names

# convert iris data set into a dataframe
import pandas as pd
df = pd.DataFrame(iris.data, columns = iris.feature_names)
df.head()

# add target column also to df
df['target'] = iris.target
df.head()

# to display pair plots
import matplotlib.pyplot as plt
import seaborn as sns

# pair plots. hue='target' -> change color based on target flower
# this use blue, red and green palette
sns.pairplot(df, hue = 'target', palette = "brg")
plt.show()

# from the above graphs , we can understand that
# we can separate the datapoints using a line
# x represnts all columns except the target column
x = df.drop(['target'], axis = 'columns')
# y represents the target column
y = df.target

# let us split data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

# train the svm model
from sklearn.svm import svm
model = SVC() # C = 0.1, kernel = 'rbf', gamma = 'scale'
model.fit(x_train, y_train)

# find accuracy level
model.score(x_test, y_test) # 0.955555

# take a random sample for prediction in the data 0 to 49 = setosa, 50 to 59 = versicolor, 100 to 149 - virginica
model.predict([[7.7, 2.6, 6.9, 2.3]]) # array([2]) -> virginica