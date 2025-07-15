Apply logistic Regression model to analyze the data to know why the employees are leaving company and when they will stay back in the company

# logistic regression - binary classification
import pandas as pd

# load the data into dataframe
df = pd.read_csv("E:/test/HR_comma_sep.csv")
df.head()

# data exploration
# find no. of rows and cols who left (i.e, left = 1) and retained (left=0)
left = df[df.left == 1]
left.shape # (3571, 10)

retained = df[df.left == 0]
retained.shape # (11428, 10)

# find averages separately for left and retained people on all columns
average = df.groupby('left').mean()

# since there is vast difference , the following are significant:
# satisfaction - level, average_monthly_hours, promotion_last_5_years,

# bar chart to know impact of employee salary
# below bar chart shows employees with high salary are not leaving (very less)
pd.crosstab(df.dept, df.left).plot(kind = 'bar')

# bar chart showing relation between dept and left
# output bar graph doesnot show significant left in terms of department
pd.Crosstab(df.sales, df.left).plot(kind = 'bar')

# from the above data analysis we can conclude that we will use following variables as dependent variables in our model
# 1. Satisfaction Level , 2. Average monthly hours, 3.Promotion last 5years , 4. salary
subdf = df[['satisfaction_level', 'average_monthly_hours', 'promotion_last_5years',' salary']]
subdf.head()

# since salary is text, we will convert it into numbers.
# we should split salary variable into salary_high
# salary_low, salary_medium. for this purpose we use dummy variables.
salary_dummies = pd.get_dummies(subdf.salary, prefix="salary")
salary_dummies

# add the dummy variable to the data frame
df_with_dummies = pd.concat([subdf, salary_dummies], axis = 'columns')
df_with_dummies.head()

# to avoid dummy variable trap, let us delete salary_medium column
# we will also remove salary column as it is already represented by dummies
df_with_dummies.drop(['salary', 'salary_dummies'], axis = 'columns', inplace = True)
df_with_dummies.head()

# take independent features (x) anf target feature (y)
x = df_with_dummies
y = df.left

# split the data into 70% train data and remaining 30% test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7)

# apply logistic regression on the train data
from sklearn.linear_model import LogisticRegression
model = logisticRegression()
model.fit(x_train, y_train)

# accuracy of our model
model.score(x_test, y_test) # 0.7733

# find if the employee will leave or stay back in the company when satisfication level is 0.11, average monthly hours in 286,
# no promotion in last 5years and medium salary (0,0).
input = [[0.11, 286, 0,0,0]]
model.predict(inputs) # array [1] will leave

# find if the employee will leave or stay back in the company if satisfication level is 0.48 average monthly hours is 228,
# no promotion in last 5 years and low salary (0,1)

inputs = [[0.48, 228,0,0,1]]
model.predict(inputs) # array[0] will stay in the company

