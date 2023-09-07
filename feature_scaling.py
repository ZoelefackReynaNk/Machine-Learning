# from sklearn.preprocessing import Imputer
# import sklearn as sk

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing the data set
data = pd.read_csv('data.csv')
# print(data.head())

# seperating dataset into dependent and independent variable arrays
X = data.iloc[:, :-1].values
Y = data.iloc[:, 3].values
print(data, "\n")
print(X, "\n")
print(Y, "\n")

# taking care of missing data
imputer1 = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer1.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print(X, "\n")

# categorisation of the data set
labelencoder_x = LabelEncoder()
X[:, 0] = labelencoder_x.fit_transform(X[:, 0])
print(X, "\n")

# dummy Encoding
# onehotencoder = OneHotEncoder(categories=[0])
# X = onehotencoder.fit_transform(X).
# print(X)

# ColumnTransformer to specify the column index not categories parameter
# The column numbers to be transformed (here is [0] but can be [0, 1, 3])
# # Leave the rest of the columns untouched
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(
    categories='auto'), [0])], remainder='passthrough')
X = ct.fit_transform(X)
print(X, "\n")
# machine learning model identifies
# dependent variables already as a category
# no need for the onehot encoder class.
labelencoder_y = LabelEncoder()
Y = labelencoder_y.fit_transform(Y)
print(Y, "\n")


# spitting the dataset into training and test data sets
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)
print(x_train, "\n\n", x_test, "\n\n", y_train, "\n\n", y_test, "\n\n")

# feature scaling
SS_X = StandardScaler()
x_train = SS_X.fit_transform(x_train)
x_test = SS_X.fit_transform(x_test)
print(x_train, "\n\n", x_test, "\n\n")
