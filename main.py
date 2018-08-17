# Titanic: Machine Learning from Disaster

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
train_dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')

# Making the matix of features and result for training data
X = train_dataset.iloc[:, 4].values
y = train_dataset.iloc[:, 1].values

# Making matrix of features and result for test data
X_test = test_dataset.iloc[:, 3].values

# Handling the categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoder = LabelEncoder()
X = encoder.fit_transform(X)
hot_encoder = OneHotEncoder(categorical_features=[0])
X = hot_encoder.fit_transform(X.reshape(-1, 1))

"""# Handling missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
X[:, 1:2] = imputer.fit_transform(X[:, 1].reshape(-1, 1))
X_test[:, 1:2] = imputer.fit_transform(X_test[:, 1].reshape(-1, 1))"""

# Making the random forest regressor and fiiting the training data to the regressor
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()
regressor.fit(X, y)

y_pred = regressor.predict(X_test)
y_pred[:, 0:1] = y_pred.dot(hot_encoder.active_features_).astype(float)

