import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([5, 15, 25, 35, 45, 55]).reshape(-1, 1)
y = np.array([5, 20, 14, 32, 22, 38])

model = LinearRegression()
model.fit(x, y)

r_sq = model.score(x, y)
#coefficient of determination
r_sq

#value of b0
print('slope(b0)', model.coef_)

#value of b1
print('intercept(b1)', model.intercept_)

#predicted output
y_pred = model.predict(x)
print('predicted response: ', y_pred, sep='\n')

#new inputs
x_new = np.arange(5).reshape((-1, 1))
print(x_new)

#predicted outputs
y_new = model.predict(x_new)
print(y_new)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


import matplotlib.pyplot as plt

plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_test, model.predict(x_test), color = 'blue')
plt.title('sp_attack vs speed')
plt.xlabel('sp_attack')
plt.ylabel('speed')
plt.show()
