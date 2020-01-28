import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Advertising.csv")

x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# Basic method to find the regression using all in method
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)

# using Backward Elimination
import statsmodels.formula.api as sm
x_opt = x[:, [0, 1, 2]]
regressor_ols = sm.OLS(endog=y, exog=x_opt).fit()
regressor_ols.summary()








