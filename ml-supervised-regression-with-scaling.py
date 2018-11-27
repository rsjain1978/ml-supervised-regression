from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('./marketing-unscaled.csv')
df = pd.DataFrame(dataset, columns=['Marketing Spend','Revenue'])

X = df[['Marketing Spend']]
Y = df[['Revenue']]

X=X.values.reshape(-1,1)
#split the data as training and test data.
train_X, test_X, train_Y, test_Y = train_test_split(X,Y, test_size=0.2)

#initialize scaler object
scaler = StandardScaler()

#scale the training X data
train_scaled_X = scaler.fit_transform(train_X)

#scale the training Y data
train_scaled_Y = scaler.transform(train_Y)

regressor = LinearRegression()
regressor.fit(train_scaled_X, train_scaled_Y)

print (regressor.coef_)

#check the predicted value from testing X data
for testX in test_X:
    testX = np.asscalar(testX)
    print ('X='+format(testX)+'\t Y^=%2f'%regressor.predict(testX))