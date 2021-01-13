# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 11:08:14 2021

@author: user
"""

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

from sklearn.preprocessing  import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()
X[:, 3]=labelencoder_X.fit_transform(X[:, 3])
ct= ColumnTransformer([("State", OneHotEncoder(),[3])], remainder='passthrough')
X=ct.fit_transform(X)

#avoiding Dummy variable
X = X[:, 1:]
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train, y_train)

y_pred=regressor.predict(X_test)

import statsmodels.api as sm
X= np.append(arr=np.ones((50,1)).astype(int), values=X, axis=1)

X_opt = np.array(X[:, [0, 3]], dtype=float)
regressor_OLS=sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary())

