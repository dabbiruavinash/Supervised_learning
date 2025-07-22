A program to categorize the given news text into one of the avaliable. 20 categories of news groups, using Multinomial Naive Bayes Machine Learning Model.

# import 20 news groups dataset
from sklearn.datasets import fetch_20newsgroups
data = fetch_20newsgroups()

# display the 20 news groups names
data.target_names

# collect train data and test data for the chosen cateogories
train = fetch_20newsgroups(subset = 'train')
test = fetch_20newsgroups(subset = 'test')

# what are the types of train and test objects?
# they are bunch objects
type(train)
type(test)

# As a sample, display 5th row from train data
print(train.data[5])

# see the category number to which the above data belongs.
print(train.target[5]) # 16

# create count vectorizer
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

# convert train and test data into numeric
traincv = cv.fit_transform(train.data)
testcv = cv.transform(test.data)

# create Multinomial Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()

# train the model
mnb.fit(traincv, train.target)

# accuracy of our model
mnb.score(testcv, test.target) # 0.7728359001593202

# let us predict the categories of our own messages
messages = ['Jesus Christ', 'apply brake at the right of your bike', 'how to send a satellite into the space']

# convert these messages into numeric using count vectorizer
test_data = cv.transform(messages)

# call predict() method
pred = mnb.predict(test_data)

# display the results
pred # array([15,8,14])