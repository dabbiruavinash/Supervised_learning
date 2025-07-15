we are given data like age and brought_insurance. Apply Logistic Regression Model and predict wheather a person takes insurance or not based on his age.

# logistic regression - binary classification
import pandas as pd
import matplotlib.pyplot as plt

# load the data into dataframe
df = pd.read_csv("E:/test/insurance_data.csv")
df

# retrieve the data
x = df.iloc[:, :-1].values # retireve only 0th column
x

y = df.iloc[:,1].values # retrieve 1st column
y

# display the scatter plot to know how the datapoints are aligned
plt.xlabel('Age')
plt.ylabel('Have insurance?')
plt.scatter(x,y,marker='+', color='red')

# for the above data, we cannot use linear regression
# show the data as logistic regression plot
import seaborn as sns
sns.regplot(x='age', y='bought_insurance', data='df, logistic=True, marker='+', color='red')

# create logistic regression model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

# train the model
model.fit(x,y)

# find accuracy of the model - gives 88.8%
model.score(x,y)

# predict if 56 years aged person will buy insurance or not
model.predict([[56]]) # array([1]) yes

# predict if 36years aged person will buy insurance or not
model.predict([[36]]) # array([0]) No