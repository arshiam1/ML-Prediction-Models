import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
import math
from scipy import stats
import csv
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing
from sklearn import utils
import random
%matplotlib inline

#Reading The Data
df = pd.read_csv("Real estate.csv")
df.head()

#Visualizing the Data
sns.heatmap(df.corr(), annot=True)

#Data Seperation as X and y
X=df.drop('Y house price of unit area',axis=1)
y=df['Y house price of unit area'] #target variable
X.head(4)

#Data Splitting
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state =100)

Model Building (Linear Regression)
"""Training the Data"""
lr = LinearRegression()
lr.fit(X_train, y_train)

"""Aplying the model to make a prediction"""
y_lr_train_pred = lr.predict(X_train)
y_lr_test_pred = lr.predict(X_test)
pd.DataFrame({'Test': y_test,'Pred':y_lr_test_pred }).head(5)

#Evaluating The Model
from sklearn.metrics import mean_squared_error, r2_score
lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)
lr_results = pd.DataFrame(['Linear Regression',lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']
lr_results


#Data Visualization and Predicted Results
plt.figure(figsize=(5,5))
plt.scatter(x = y_train, y = y_lr_train_pred, color = 'purple', alpha =0.3)
z= np.polyfit(y_train, y_lr_train_pred, 1)
p = np.poly1d(z)
plt.plot(y_train, p(y_train), color = 'blue')
plt.ylabel('Predicted house price of unit area')
plt.xlabel('Experimental house price of unit area')


