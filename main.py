import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor

ds = pd.read_csv('Position_Salaries.csv')

X = ds.iloc[:, 1:-1].values
y = ds.iloc[:, -1].values

reg = DecisionTreeRegressor(random_state=0)
reg.fit(X, y)


X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, reg.predict(X_grid), color='blue')
plt.title('Decision Tree Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
