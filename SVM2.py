Read data from hand written digits data set. Apply SVM on it and analyse the data. Observe the accuracy of the model with different kernels. Then predict the digits from the images of hand written digits.

# hand written digits classification using svm
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
digits = load_digits()

# display attributes of the digits object
dir(digits)

# display data of image 0 - shows an array
digits.data[0]

# display target (or original) digit - shows 0
digits.target[0]

# let us split the digits.data into train and test data
# take digits.data as independent var and digits.target as target var
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(digits.data, digits.target, test_size = 0.2)

# create SVC class object and train it
from sklearn.svm import SVC

# C = 0.1, kernel = 'rbf', gamma = 'scale'
model = SVC()
model.fit(x_train, y_train)
model.score(x_test, y_test) # 0.9805555555555555

# C = 0.1, kernel = 'linear', gamma = 'scale'
model = SVC(kernel = 'linear')
model.fit(x_train, y_train)
model.score(x_test, y_test) # 0.9694444444444444

# C = 0.1, kernel = 'sigmoid', gamma = 'scale'
model = SVC(kernel = 'sigmoid')
model.fit(x_train, y_train)
model.score(x_test, y_test) # 0.90555555555555555

# C = 0.1, kernel = 'poly', gamma = 'scale'
model = SVC(kernel = 'poly')
model.fit(x_train, y_train)
model.score(x_test, y_test) # 0.98611111111111112

# C = 1.0, kernel = 'poly', gamma = 'auto'
model = SVC(kernel = 'poly', gamma = 'auto')
model.fit(x_train, y_train)
model.score(x_test, y_test) # 0.98611111111111111

# take a sample from test data - take 100th row
x_test[100]

# predict which digit it is 
model.predict([x_test[100]]) # array([9])

# compare it with actual digit - it is
y_test[100] # it is 9

# another way of predicting the digit directly form digits data set.
# let us take 67th row from digits object
digits.data[67] # array for 67th image

# this is the 67th image 0 - looks like 6
plt.matshow(digits.images[67])

# let us predict which image it is - gives 6
model.predict([digit.data[67]]) # array([6]) # array([6])

# compare it with actual target - it is 6
digits.target[67]