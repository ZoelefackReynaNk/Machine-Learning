# from sklearn.preprocessing import Imputer
# import sklearn as sk
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')
# print(data.head())

# seperating dataset into dependent and independent variable arrays
X = data.iloc[:, :-1].values
Y = data.iloc[:, 3].values
print(data, "\n")
print(X, "\n")
print(Y)

# taking care of missing data
imputer = SimpleImputer(missing_values='NaN', strategy='mean')
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform[X[:, 1:3]]
print(X)
