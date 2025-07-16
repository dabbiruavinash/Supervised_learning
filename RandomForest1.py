Train the Random Forest on the scikit digits datasets and check if the model is correctly predicting the handwritten digits

# random forest classifier -- recognizing handwritten digits
# load the digits dataset
from sklearn.datasets import load_digits
digits = load_digits()

# see the column names in the dataset
dir(digits)

# digits.images - - -> array of images. each image is of 8x8 pixels
# digits.data - - -> array of data related to images. Each array is of 64 values
# digits.target - - -> values digit representing the image

# display the first 10 digits images
import matplotlib.pyplot as plt
plt.gray() # show in gray color
     plt.matshow(digits.images[i])

# create dataframe with all data
import pandas as pd
df = pd.DataFrame(digits.data)
df.head()

#in the above output, the 1st row -> 0, 2nd row -> 1 etc add target data to dataframe
df['target'] = digits.target
df.head()

# take input data as x and target as y
x = df.drop(['target'], axis = 'columns')
y = df['target].values

# split the data as train and test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

# the word ensemble indicates using multiple algorithms (Decision Tree) to predict output
# default no of random trees = n_estimators = 100
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train, y_train)

# what is the score with 100 trees ?
model.score(x_test, y_test) # 0.9777777777777

# what is the score with 50 trees?
model = RandomForestClassifier(n_estimators=50)
model.fit(x_train, y_train)
model.score(x_test, y_test) # 0.975

# make prediction
# find out the handwritten digit contained in 12 th row in data
model.predict([digits.data[12]]) # array([2])

# display its image to verify
plt.matshow(digits.image[12]) # seems to be 2

# match with the target that shows original digit
print(digits.target[12]) # 2