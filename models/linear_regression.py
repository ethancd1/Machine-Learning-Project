## Header

## Imports

import numpy as np 
from sklearn.linear_model import LinearRegression

x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])

model = LinearRegression()

model.fit(x,y)
# LinearRegression()

model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print(f"coefficient of determination: {r_sq}")
# coefficient of determination: 0.7158756137479542

print(f"intercept: {model.intercept_}")
# intercept: 5.633333333333329

print(f"slope: {model.coef_}")
# slope: [0.54]

new_model = LinearRegression().fit(x, y.reshape((-1, 1)))
print(f"intercept: {new_model.intercept_}")
# intercept: [5.63333333]

print(f"slope: {new_model.coef_}")
# slope: [[0.54]]