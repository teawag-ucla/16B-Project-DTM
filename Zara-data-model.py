# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 16:43:36 2025

@author: TeaWagstaff

this code was moved here from my jupyter notebook (easier to test code there)
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from pca import Standard_PCA

df_og = pd.read_csv('Zara_sales_EDA.csv', sep=';')
df_og.head()

#preprocessing categories into numbers, and dropping the columns we won'd be using

df=df_og.copy()
df = pd.get_dummies(df, columns=['Product Position', 'Promotion', 'Seasonal', 'terms', 'section', 'season', 'material', 'origin'])

df.drop('Product Category', axis=1, inplace=True)
df.drop('brand', axis=1, inplace=True)
df.drop('url', axis=1, inplace=True)
df.drop('name', axis=1, inplace=True)
df.drop('description', axis=1, inplace=True)
df.drop('currency', axis=1, inplace=True)

df = df.dropna()
correlation_matrix = df.corr()
#this plot can't show all columns, but gives some interesting information
sns.heatmap(correlation_matrix)
plt.title('Zara Sales Data Correlation Matrix')
plt.show()

X = df.copy().drop('Sales Volume', axis=1)
y = df['Sales Volume']


#pca and model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#df has 44 columns, so we should knock it down a bit
#we will choose how many components to have using the following loop
for i in range(1, 43):
    pca = Standard_PCA(n_components = i)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    explained_variance_ratio = pca.pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    print(i, cumulative_variance_ratio)

#we can see that 100% of the total variance can be explained by 35 columns and 80% can be explained by 26
#so let's use n_components = 26

pca = Standard_PCA(n_components = 26)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

regressor = DecisionTreeRegressor(random_state=0, max_depth=10)
regressor.fit(X_train, y_train)
print(regressor.score(X_test, y_test))


regressor_pca = DecisionTreeRegressor(random_state=0,max_depth=10)
regressor_pca.fit(X_train_pca, y_train)
print(regressor_pca.score(X_test_pca, y_test))

#this section shows how our pca tree can be more accurate than the original data