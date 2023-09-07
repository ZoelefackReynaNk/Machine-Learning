# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

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