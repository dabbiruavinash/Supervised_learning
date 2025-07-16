Create a python program using a decision tree machine learning model to analyse employee salary data of various companies and then predict the salary of a new employee.

# prediction for salary of an employee -- decision tree
import pandas as pd

# load the dataset
df = pd.read_csv('e:/test/salaries.csv')
df

# drop the target column and take others as input
input = df.drop('salary_more_than_100k', axis = 'columns')

# take only target column separatly
target = df['salary_more_than_100k']

# let su convert the column data into numeric this is done with LabelEncoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

' ' ' Encoding done by LabelEncoder:
     company - - -> 0 amazon, 1 jp morgan, 2 microsoft
     job - - -> 0 programmer, 1 project manager, 2 sales executive
     degree - - -> 0 bachelors, 1 masters
' ' '

inputs['company'] = le.fit_transform(inputs['company'])
inputs['job_n'] = le.fit_transform(input['job'])
inputs['degree_n'] = le.fit_transform(inputs['degree'])
inputs

# delete cols with labels (or Strings)
# keep only cols with numeric values
inputs_n = inputs.drop(['company' , 'job', 'degree'], axis = 'columns')
inputs_n

# create the model
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(input_n, target)

# predict for a person working in google as sales executive with master degree
model.predict([[1,1,0]]) # array([0]) # less than 10k
model.predict([[2,0,0]]) # array([0]) # >= 100k