Take the SMS spam collections dataset and analyze which messages are spam and which are ham by creating a spam filter using Multinomial Navie Bayes machine learning model
Apply this spam filter to predict the given messages are spam or ham

# multinomial Navies Bayes Classification - spam filter
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# load the dataset
df = pd.read_csv('E:/test/smsspamcollection/SMSSpamCollection', sep = '\t', names = ['Status', 'Message'])
df.head()

# find how many spam records and how many are not spam (i.e, ham)
len(df[df.Status == 'spam']) # 747
len(df[df.Status == 'ham]) # 4825

# convert spam as 0 and ham as 1
df.loc[df["Status"] == 'spam', "Status] = 0
df.loc[df["Status"] == 'ham', "Status"] = 1
df.head()

# Messages are our data and Status is what we have to predict
x = df["Message"]
y = df["Status"]

# split the data into trainng the testing parts
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)

# count vectorizer counts the frequency of words and gives a matrix of frequencies
x_traincv = cv.fit_trainsform(x_train)
y_traincv = cv.transform(x_test)

# before we train the model, take y as int type
y_train = y_train.astype('int')
y_test = y_test.astype('int')

# train the model
mnb = MultinomialNB()
mnb.fit(x_traincv, y_train)

# find accuracy of the model
mnb.score(x_testcv,y_test) # 0.9838565022421525

# take an sms and predict if it is spam or ham
x_test.iloc[0]
pred = mnb.predict(x_testcv[0])
pred # array([1]) that means it is not spam

# take another sms and predict if it is spam or ham
x_test.iloc[11]
pred = mnb.predict(x_testcv[11])
pred # array([0]_ that means it is spam

# predict for our own message if it is spam or ham
x_test.iloc[0]
pred = mnb.predict(x_testcv[0])
pred # array([1]) that means it is not spam

# take another sms and predict if it is spam or ham
x_test.iloc[11]
pred = mnb.predict(x_testcv[11])
pred # array([0]) that means is it spam

# predict for our own messages
# output 0 -> spam and 1 -> ham
examples = ['Free Viagra now', 'Hi, can we play golf tomorrow?']
test = cv.transform(test)
pred = mnb.predict(test)
pred = array[0,1]


