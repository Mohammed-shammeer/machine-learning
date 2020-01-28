import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

dataset = pd.read_csv('USA_51.csv')

x = dataset.iloc[:, 2].values
y = dataset.iloc[:, 3].values

x= x.reshape(-1, 1)
y= y.reshape(-1, 1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

x_pred = regressor.predict(x_train)

y_pred = regressor.intercept_+regressor.coef_*x_train

plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Birthrate')
plt.xlabel('15 to 16')
plt.ylabel('17 to 18')
plt.show()

import statsmodels.api as sm
X2 = sm.add_constant(x)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())