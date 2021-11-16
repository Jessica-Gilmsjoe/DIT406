#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 13:14:38 2021

@author: jessicagilmsjoe
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay

#%%
houses = pd.read_csv("data_assignment2.csv")

#%%
### Question 1
# a)
plt.scatter(houses[['Living_area']], houses[['Selling_price']])
plt.show()
#%%
# Model linear regression area and price
model = LinearRegression().fit(houses[['Living_area']], houses[['Selling_price']])
#%%
xfit=np.linspace(min(houses['Living_area']),max(houses['Living_area']), 1000) #1000 evenly spaced points in [0, 55].
yfit=model.predict(xfit[:, np.newaxis])
plt.scatter(houses[['Living_area']], houses[['Selling_price']])
plt.plot(xfit, yfit)
plt.show()

#%%
# b)

# slope
slope = model.coef_

# intersection
intersection = model.intercept_

# c)

pred_100 = model.predict([[100]])
pred_150 = model.predict([[150]])
pred_200 = model.predict([[200]])

# d)

# residual plot
pred_price = model.predict(houses[['Living_area']])
residuals = pred_price - houses[['Selling_price']]
plt.scatter(pred_price, residuals, c = 'b', s = 50, alpha = 0.4)
plt.hlines(y = 0, xmin = 3400000, xmax = 6500000)
plt.title('Residual plot')
plt.ylabel('Residuals')

#%%
### Question 2

# a)
# load iris data and create dataframe
from sklearn.datasets import load_iris
iris_raw = load_iris()

iris = pd.DataFrame(iris_raw.data , columns = iris_raw.feature_names)
iris['species'] = iris_raw.target 
iris['species'] = iris['species'].replace(to_replace= [0, 1, 2], 
    value = ['setosa', 'versicolor', 'virginica'])

sns.FacetGrid(iris, hue ="species",
              height = 6).map(plt.scatter,
                              'sepal length (cm)',
                              'petal length (cm)').add_legend()
#%%
x_train, x_test, y_train, y_test = train_test_split(iris[iris.columns[0:4]], 
                                                    iris[['species']], 
                                                    test_size=0.40, random_state=0)

model = LogisticRegression().fit(x_train, y_train)

fig = plt.figure()
plot_confusion_matrix(model,x_test,y_test, cmap='Blues')
plt.title('Confusion matrix logistic regression')

y_pred=model.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#%%
# b)
## knn
k = 5
knn = KNeighborsClassifier(n_neighbors=k, weights = 'uniform')
knn.fit(x_train, y_train) 

#%%
y_pred=knn.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#%%
plot_confusion_matrix(knn,x_test,y_test, cmap='Blues')
       
#%%
pred = knn.predict(x_test)
cm = confusion_matrix(y_test, pred)

ax = sns.heatmap(cm, annot=True, cmap='Blues')

ax.set_title('Confusion Matrix Knn\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

ax.xaxis.set_ticklabels(['Setosa','Vercicolor', 'Virginica'])
ax.yaxis.set_ticklabels(['Setosa','Vercicolor', 'Virginica'])

plt.show()

#%%
classifiers = [KNeighborsClassifier(n_neighbors=1, weights = 'uniform'), 
               KNeighborsClassifier(n_neighbors=5, weights = 'uniform'),
               KNeighborsClassifier(n_neighbors=10, weights = 'uniform'), 
               KNeighborsClassifier(n_neighbors=20, weights = 'uniform')]

for cls in classifiers:
    cls.fit(x_train, y_train)
    
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,10))

for cls, ax in zip(classifiers, axes.flatten()):
    plot_confusion_matrix(cls, 
                          x_test, 
                          y_test, 
                          ax=ax, 
                          cmap='Blues',
                         display_labels=['Setosa','Vercicolor', 'Virginica'])
    ax.set_title('Neighbors = '+str(cls.n_neighbors))
plt.tight_layout()  
plt.show()

#%%
classifiers = [KNeighborsClassifier(n_neighbors=1, weights = 'distance'), 
               KNeighborsClassifier(n_neighbors=5, weights = 'distance'),
               KNeighborsClassifier(n_neighbors=10, weights = 'distance'), 
               KNeighborsClassifier(n_neighbors=20, weights = 'distance')]

for cls in classifiers:
    cls.fit(x_train, y_train)
    
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,10))

for cls, ax in zip(classifiers, axes.flatten()):
    plot_confusion_matrix(cls, 
                          x_test, 
                          y_test, 
                          ax=ax, 
                          cmap='Blues',
                         display_labels=['Setosa','Vercicolor', 'Virginica'])
    ax.set_title('Neighbors = '+str(cls.n_neighbors))
plt.tight_layout()  
plt.show()

