import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Run file with command: python -m src.zalmoxis.lin_regr

# Given data points
x = np.array([10, 15, 16]).reshape(-1, 1)
y = np.array([0.501290, 0.501925, 0.502042])

# Perform linear regression
model = LinearRegression()
model.fit(x, y)
r_squared = model.score(x, y)
print(f'R^2 value: {r_squared}')
model.fit(x, y)

# Extrapolate for values from 1 to 50
x_new = np.arange(1, 51).reshape(-1, 1)
y_new = model.predict(x_new)

# Print x_new, y_new pairs
for x_val, y_val in zip(x_new, y_new):
    print(f'x: {x_val[0]:.6f}, y: {y_val:.6f}')

# Plot the original data points
plt.scatter(x, y, color='red', label='Data Points')

# Plot the linear regression line
plt.plot(x_new, y_new, color='blue', label='Linear Regression')

# Add labels and legend
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Linear Regression and Extrapolation')
plt.legend()

# Show the plot
#plt.show()