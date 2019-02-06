# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 23:22:09 2017

@author: Tin
"""
# Porto Seguro’s Safe Driver Prediction
# Description: Predict if a Driver will file an insurance claim next year (Define Problem)
# Three Dataset: train.csv, train.csv, sample_submission.csv  

# Libraries
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt # for plotting
%matplotlib inline

import time
import warnings
warnings.filterwarnings('ignore')

Train_set = pd.read_csv('C:/Users/Tin Hang/Desktop/TestPythonCodes/DriverPrediction/train.csv')
Test_set = pd.read_csv('C:/Users/Tin Hang/Desktop/TestPythonCodes/DriverPrediction/test.csv')
Sample_set = pd.read_csv("C:/Users/Tin Hang/Desktop/TestPythonCodes/DriverPrediction/sample_submission.csv")

from sklearn import model_selection
from sklearn import preprocessing, cross_validation

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC

# multivariate plot
# scatter plot matrix
scatter_matrix(Train_set)
plt.show()

# Split-out validation dataset
array = Train_set.values
X = array[:,0:58]
Y = array[:,58]
validation_size = 0.15
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
seed = 7

scoring = 'accuracy'


# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('RF', RandomForestClassifier()))
models.append(('MLP', MLPClassifier()))
models.append(('SVMG', SGDClassifier()))
models.append(('SVML', LinearSVC()))
models.append(('SVM', SVC()))


# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	%time
print(msg)


import time
start_time = time.time()
main()
print("--- %s seconds ---" % (time.time() - start_time))

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Make predictions on validation dataset
svm = SVC()
svm.fit(X_train, Y_train)
predictions = svm.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# Importance Features
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

y = target
X = Train_set

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)
clf = RandomForestRegressor(n_jobs=2, n_estimators=1000)
model = clf.fit(X_train, y_train)

from tabulate import tabulate
headers = ["Features:", "Score:"]
values = sorted(zip(X_train.columns, model.feature_importances_), key=lambda x: x[1] * -1)
print(tabulate(values, headers, tablefmt="plain"))



c_features = len(Train_set)
plt.barh(range(c_features), clf.feature_importances_)
plt.xlabel("Feature importance")
plt.ylabel("Feature name")
plt.yticks(numpy.arange(c_features), feature_names)
plt.figure(figsize=(16,16))
plt.show()

print('Feature importances: {}'
      .format(clf.feature_importances_))



# Univariate Statistical - Select features that has the best relationships with the output variable 
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X = Train_set[:,0:56]
Y = Train_set[:,56]
# feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, Y)
# summarize scores
set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
# summarize selected features
print(features[0:56,:])


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# Recursive Feature Elimination
X = Train_set[:,0:55]
Y = Train_set[:,55]
# feature extraction
model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)
print("Num Features: %d") % fit.n_features_
print("Selected Features: %s") % fit.support_
print("Feature Ranking: %s") % fit.ranking_


# Principal Component Analysis
from sklearn.decomposition import PCA

X = array[:,0:55]
Y = array[:,55]
# feature extraction
pca = PCA(n_components=3)
fit = pca.fit(X)
# summarize components
print("Explained Variance: %s") % fit.explained_variance_ratio_
print(fit.components_)


# Select the score for each features
from sklearn.ensemble import ExtraTreesClassifier

X = array[:,0:55]
Y = array[:,55]
# feature extraction
model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)


