# -*- coding: utf-8 -*-
"""

"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

iris=pd.read_csv("C:/Users/ADMIN/Downloads/Iris.csv")
iris.head()
iris.drop("Id", axis=1, inplace=True)

#eda
iris.info()
iris.describe()
iris.columns

#checking missing values
iris.isna().sum()

#checking for outliers
import matplotlib.pyplot as plt
plt.boxplot(iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']])
#no outliers found

#data visualization
import seaborn as sns
#scatter plot for sepal lemgth and sepal width
sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=iris, size=5)

#adding species name
sns.FacetGrid(iris, hue="Species", palette="husl", size=5) \
   .map(plt.scatter, "SepalLengthCm", "SepalWidthCm") \
   .add_legend()

#identifying the specie with the highest petal length 
sns.FacetGrid(iris, hue="Species", palette="husl", size=6).map(sns.kdeplot, "PetalLengthCm") \
   .add_legend()
#iris setosa

#identifying the specie with highest sepal_width
sns.FacetGrid(iris,hue='Species', palette='husl', size=6).map(sns.kdeplot, 'SepalWidthCm')\
    .add_legend()
#iris-verginica

#identifying the specie with highest sepal length
sns.FacetGrid(iris,hue='Species', palette='husl', size=6).map(sns.kdeplot, 'SepalLengthCm')\
    .add_legend()
#iris setosa

#identifying the specie with highest petal width
sns.FacetGrid(iris,hue='Species', palette='husl', size=6).map(sns.kdeplot, 'PetalWidthCm')\
    .add_legend()
#iris setosa
#pairplot
sns.pairplot(iris, hue="Species", palette="husl", size=3, diag_kind="kde")

#Model Making
X=iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y=iris['Species']

#splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) 

#Randomn Forest
#creation of model
from sklearn.ensemble import RandomForestClassifier 
clf=RandomForestClassifier(n_estimators=100)

#fitting a model
clf.fit(X_train,y_train)

#prediction
y_pred=clf.predict(X_test)

from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#accuracy=0.9555

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier()
#fitting a model
tree.fit(X_train,y_train)

#prediction
prediction=tree.predict(X_test)

#accuracy
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(prediction,y_test))

#accuracy = 0.9333

#knn
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=8)

#fitting the model
classifier.fit(X_train, y_train)

#prediction
predict = classifier.predict(X_test)

#accuracy
print('accuracy is',accuracy_score(predict,y_test))

#accuracy=0.9555

#Conclusion - All the three models are good for the current dtaa














































