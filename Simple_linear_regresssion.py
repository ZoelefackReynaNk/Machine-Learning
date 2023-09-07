# Data Preprocessing Template

# Importing the libraries
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

print(f" X = {X} \n\n Y = {Y} \n\n")

# Splitting the dataset into the Training set and Test set
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=1/3, random_state=0)
print(
    f"X_train = {X_train} \n\n X_test = {X_test} \n\n Y_train = {Y_train} \n\n Y_test = {Y_test} \n\n")

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""

# fitting Simple Linear Regression to the training set.
regressor = LinearRegression()
fitted = regressor.fit(X_train, Y_train)


# Predicting the test set results
Y_pred = regressor.predict(X_test)
print(f"Y_pred = {Y_pred} \n\n")

# visualising the training set results
# 1. real observation point: number of years of experience(X) vs salary(Y)
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience[training set]')
plt.xlabel('years of experience')
plt.ylabel('Salary')
plt.show()

# visualising the test set results
# 1. real observation point: number of years of experience(X) vs salary(Y)
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience[test set]')
plt.xlabel('years of experience')
plt.ylabel('Salary')
plt.show()
