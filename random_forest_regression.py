import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

# reading the data_set
dataset = pd.read_csv('/Users/zakariadarwish/Desktop/random_forest_regression/Position_Salaries.csv')
# feeding the dependant & independant variables
# where X is dep & y is indep
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# applying Decision Tree regressor model 
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)

# predicting a single value using decision tree
y_pred = regressor.predict([[6.5]])


# displaying 
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
