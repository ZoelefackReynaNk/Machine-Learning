# Data Preprocessing Template
# Importing the libraries
import statsmodels.formula.api as sm
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values
print(f" X = {X} \n\n Y = {Y} \n\n")


'''# categorisation of the data set
labelencoder_x = LabelEncoder()
X[:, 3] = labelencoder_x.fit_transform(X[:, 3])
print(f"X_after_labelencoding = {X} \n\n")'''
# dummy Encoding
# ColumnTransformer to specify the column index
# not categories parameter.
# The column numbers to be transformed (here is [0] but can be [0, 1, 3])
# # Leave the rest of the columns untouched
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(
    categories='auto'), [3])], remainder='passthrough')
X = ct.fit_transform(X)
print(f"X_after_columntransform = {X} \n\n")


# Avoiding the dummy variable trap
X = X[:, 1:]
print(f"X_avoided = {X} \n\n")


# Splitting the dataset into the Training set and Test set
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)
print(
    f"X_train = {X_train} \n\n X_test = {X_test} \n\n Y_train = {Y_train} \n\n Y_test = {Y_test} \n\n")


# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""


# Multiple Linear regression.
# 1. Fitting multiple linear regression to the training set
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# 2. Predicting the test set results
Y_pred = regressor.predict(X_test)
print(f"y_predicted = {Y_pred} \n\n")

# 3. Building the optimal model using Backward Elimination.
X = np.append(arr=np.ones([50, 1]).astype(int), values=X, axis=1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
# ordinary least quares class.
#regressor_OLS = sm.ols(endog=Y, exog=X_opt).fit()
regressor_OLS  = sm.ols(formula='Y_variable ~ X_opt_variable', data=X_opt).fit()
#mdl = regressor_OLS.get_robustcov_results(cov_type='HAC',maxlags=1)
r = regressor_OLS.
print(r)
