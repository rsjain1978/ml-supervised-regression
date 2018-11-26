import sklearn
from sklearn.model_selection import train_test_split
from sklearn import linear_model as lm
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np
import pandas as p
import matplotlib.pyplot as plt

#read the csv file which has the fictional marketing data and create a dataframe
data=p.read_csv('./marketing.csv')

#read the value of 'marketing spend' which in an independent variable called indep
indep=data['Marketing Spend']

#read the value of 'revenue' which in our case is a dependent variable called dep
dep=data['Revenue']

#convert the value of idependent variable to a 2-D array
indep=indep.values.reshape(-1,1)

#split the independent variable data into test and training data. Here 20% of data is being set aside as test data
indep_train, indep_test, dep_train, dep_test = train_test_split(indep,dep,test_size=0.2, random_state=0)

#initialize linear regression object
regressor = lm.LinearRegression()

#fit the training data into the linear model
regressor.fit(indep_train,dep_train)

fig = plt.figure()

#draw a scatter plot which has 'training data for independent variable on X-axis' and 'predicted dependent data on Y-axis'
plt.scatter(indep_train, regressor.predict(indep_train),c='g',linestyle='--')

#draw a scatter plot which has 'training data for independent variable on X-axis' and 'actual dependent data on Y-axis'
plt.scatter(indep_train, dep_train, c='r')

plt.plot(indep_train, regressor.predict(indep_train),c='g',linewidth=3)

plt.show()

#Calculate RMSE (Root Mean Square Error)
mse=np.sqrt(mean_squared_error(dep_test, regressor.predict(indep_test)))

#Calculate variance
r2 = r2_score(dep_test, regressor.predict(indep_test))

print ('Coeffecients are:', regressor.coef_)
print ('Mean Squared Error :%2f'%mse)
print ('R2:%2f'%r2)