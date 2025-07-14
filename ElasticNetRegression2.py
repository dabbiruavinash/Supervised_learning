Write a python programming to create ElasticNet model and find out which factor is most determining factor for diabetes in Indians

# feature selection using elastic net regression
# load the diabetes dataset into df
import pandas as pd
df = pd.read_csv("E:/test/diabetes.csv")
df.head()

# delete the outcome column as it is dependent and take it as y
x = df.drop('Outcome', axis = 1)
x

y = df['Outcome'].values
y

# take the names of columns
names = x.columns
names

# range of columns - gives range(0,8)
rng = range(len(names))
rng

# create an object to ElasticNet class
from sklearn.linear_model import ElasticNet
es = ElasticNet(l1_ratio = 0.5)

# train the model
model = es.fit(x,y)

# find the coefficients of terms in elastic net model equation the term with highest coefficient will exert more influence on Outcome value
es_coef = model.coef_
es_coef

# find the position of maximum value in the coefficients 
import numpy as np
n = np.argmax(es_coef)
n # gives 1

# find the column name at nth position in x
print('The most influencing column = ', name[n])

# draw line plot between range and coefficients 
# we can see highest peak at 'Glucose' column
plt.plot(rng, es_coef)
plt.xticks(rng, names, rotation=60)
plt.ylabel("Coefficients")
plt.show()

