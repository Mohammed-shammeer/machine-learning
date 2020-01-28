import numpy as np
import scipy.interpolate as si
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('SLR_data.csv')
# x = dataset.iloc[:, 0].values
# y = dataset.iloc[:, 1].values
#
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
#
# p1 = np.polyfit(x_train, y_train, 1)
# print(p1)
#
# p2 = np.polyfit(x_train, y_train, 2)
# print(p2)
#
# p3 = np.polyfit(x_train, y_train, 3)
# print(p3)

X = dataset.iloc[:, 0:1].values
Y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
q1 = LinearRegression()
q1.fit(X_train, Y_train)

from sklearn.preprocessing import PolynomialFeatures
v1 = PolynomialFeatures(degree=2)
v1_p = v1.fit_transform(X_train)
v1.fit(v1_p, Y_train)
q2 = LinearRegression()
q2.fit(v1_p, Y_train)

v2 = PolynomialFeatures(degree=3)
v2_p = v2.fit_transform(X_train)
v2.fit(v2_p, Y_train)
q3 = LinearRegression()
q3.fit(v2_p, Y_train)

plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_test, q1.predict(X_test), color = 'blue')
# plt.show()
#
# plt.scatter(X_train, Y_train, color='red')
plt.plot(X_test, q2.predict(v1.fit_transform(X_test)), color = 'green')

# plt.show()
#
# plt.scatter(X_train, Y_train, color='red')
plt.plot(X_test, q3.predict(v2.fit_transform(X_test)), color = 'yellow')
plt.show()


# plt.scatter(x_train, y_train, color = 'red')
# plt.plot(x_train, np.polyval(p1, x_train), color = 'blue')
# # plt.show()
#
# # y_pred_p1 = np.polyval(p1, x_test)
#
# # plt.scatter(x_train, y_train, color = 'yellow')
# plt.plot(x_train, np.polyval(p2, x_train), color = 'black')
# # plt.show()
#
# # y_pred_p2 = np.polyval(p2, x_test)
#
# # plt.scatter(x_train, y_train, color = 'yellow')
# plt.plot(x_train, np.polyval(p3, x_train), color = 'green')
# plt.show()
#
# y_pred_p3 = np.polyval(p3, x_test)