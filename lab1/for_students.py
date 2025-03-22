import numpy as np
import matplotlib.pyplot as plt

from data import get_data, inspect_data, split_data

data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()

# TODO: calculate closed-form solution
ones_column = np.ones((x_train.shape[0], 1))

# Łączenie kolumny jednostek z danymi obserwacji dla cech x
x_with_bias = np.column_stack((ones_column, x_train))

# Korzystam ze wzoru 1.13 na closed-up solution
# θ = (X˙ T X˙ )^-1 * X^T * y
theta_best = np.linalg.inv(x_with_bias.T.dot(x_with_bias)).dot(x_with_bias.T).dot(y_train)

# TODO: calculate error
# Korzystam ze wzoru 1.3 MSE(θ) = 1
# 1/m * E(od 1 do m ) (theta * x - y)
MSE = 0
for i in range(np.size(x_test)):
    MSE += ((float(theta_best[0]) + float(theta_best[1]) * x_test[i]) - y_test[i])**2

MSE /= np.size(x_test)
print("ERROR poczatkowy:", MSE)

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()

# TODO: standardization
# stadnardyzacja za pomoca wzoru 1.15

mean_x_train = np.mean(x_train)
mean_y_train = np.mean(y_train)
std_x_train = np.std(x_train)
std_y_train = np.std(y_train)

x_train_z = (x_train - mean_x_train) / std_x_train
y_train_z = (y_train - mean_y_train) / std_y_train

x_test_z = (x_test - mean_x_train) / std_x_train
y_test_z = (y_test - mean_y_train) / std_y_train

# TODO: calculate theta using Batch Gradient Descent

theta = np.random.rand(2)
learning_rate = 0.01
iterations = 1000

MSE_Z = 0

for i in range(np.size(y_train)):
    MSE_Z += ((theta[0] + theta[1] * x_train_z[i]) - y_train_z[i]) ** 2\

# Dla sumy wyliczonego bledu dziele go przez ilosc zestawow
MSE_Z /= np.size(y_train)

# blad przed korzystaniem z motody gradientu prostego
print("ERROR przed :", MSE_Z)

for i in range(iterations):
    x_train_z0 = np.column_stack((np.ones_like(x_train_z), x_train_z))
    gradient_MSE = (2 / np.size(y_train)) * x_train_z0.T.dot(x_train_z0.dot(theta) - y_train_z)
    theta = theta - learning_rate * gradient_MSE

# TODO: calculate error
MSE = 0
for i in range(np.size(x_test)):
    MSE += ((float(theta[0]) + float(theta[1]) * x_test_z[i]) - y_test_z[i])**2

MSE /= np.size(x_test)
print("ERROR koncowy :", MSE)

# plot the regression line
x = np.linspace(min(x_test_z), max(x_test_z), 100)
y = float(theta[0]) + float(theta[1]) * x
plt.plot(x, y)
plt.scatter(x_test_z, y_test_z)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()