# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 21:14:31 2017

@author: Tin
"""

# Libraries
import pandas as pd
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib as mpl
import seaborn as sns
import missingno as msno
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier

import warnings
from collections import Counter
warnings.filterwarnings('ignore')

train_set = pd.read_csv('C:/Users/Tin Hang/Desktop/TestPythonCodes/DriverPrediction/train.csv')
test_set = pd.read_csv('C:/Users/Tin Hang/Desktop/TestPythonCodes/DriverPrediction/test.csv')
Sample_set = pd.read_csv("C:/Users/Tin Hang/Desktop/TestPythonCodes/DriverPrediction/sample_submission.csv")


# Analyze Data - Descriptive statistics and visualization
# Data integration, selection, cleaning and pre-processing
# Read train_set and test_set Data
train_set.head() 
train_set.tail()
test_set.head() 
test_set.tail()

train_set.info() # 595212 rows and 59 Columns
train_set.dtypes
train_set.shape
train_set.describe() 
train_set.columns
train_set.isnull().sum() # check missing values

test_set.info() 
test_set.dtypes
test_set.shape
test_set.describe() 
test_set.columns
test_set.isnull().sum() # check missing values

#Obseveration on Memory Usage
train_set.info(memory_usage='deep',verbose=False)
test_set.info(memory_usage='deep',verbose=False)

# Missing Data
print("Train Missing Data: %d" %train_set.isnull().sum().sum())
print("Test Missing Data: %d" %test_set.isnull().sum().sum())

train_missing_count = (train_set == -1).sum()
plt.rcParams['figure.figsize'] = (15,8)
train_missing_count.plot.bar()
plt.show()

test_missing_count = (test_set == -1).sum()
test_missing_count.plot.bar()
plt.show()

# Display Report
from pandas_profiling import ProfileReport

profile1 = ProfileReport(train_set)
profile1.to_file(outputfile="C:/Users/Tin Hang/Desktop/TestPythonCodes/DriverPrediction/train_set_data.html")

profile2 = ProfileReport(test_set)
profile2.to_file(outputfile="C:/Users/Tin Hang/Desktop/TestPythonCodes/DriverPrediction/test_set_data.html")


# Analysis Features
unique_counter = Counter()
for col in train_set.columns:
    unique_counter[col] = len(np.sort(train_set[col].unique()))
binary_columns = [ col for col , val in unique_counter.items() if(val==2)]
binary_column_sum = []
for col in binary_columns:
    binary_column_sum.append(train_set[col].sum())
#List of binary columns
binary_columns

# data to plot
n_groups = len(binary_columns)
one_cols = binary_column_sum
zero_cols = train_set.shape[0] - np.asarray(binary_column_sum)
 
# create plot
plt.rcParams['figure.figsize'] = (15,8)
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.3
opacity = 0.8
 
rects1 = plt.bar(index, one_cols, bar_width,
                 alpha=opacity,
                 color='g',
                 label='1')
 
rects2 = plt.bar(index + bar_width, zero_cols, bar_width,
                 alpha=opacity,
                 color='b',
                 label='0')

plt.ylabel('#', fontsize=14)
plt.title('Binary Features', fontsize=20)
plt.xticks(index + bar_width/2, binary_columns, rotation='vertical', fontsize=12)
plt.legend()
 
plt.tight_layout()
plt.show()


# Catagorical and other Features
# Univariate Histograms
columns_multi = [x for x in list(train_set.columns) if x not in binary_columns]
columns_multi.remove('id')
columns_multi
plt.rcParams['figure.figsize'] = (15,40)
names = columns_multi
train_set.hist(layout = (10,4), column = columns_multi)
plt.show()

names = columns_multi
train_set.plot(kind='density', subplots=True, layout=(15,4), sharex=False)
plt.show()

# Important Features
# Create a random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
# Train the classifier
clf.fit(train.iloc[:,2:], train.iloc[:,1])

# Print the name and gini importance of each feature
feature_importances = sorted(zip(clf.feature_importances_, list(train.columns)[2:]), reverse=True)
objects = (list(zip(*feature_importances)[1]))
y_pos = np.arange(len(objects))
performance = np.array(zip(*feature_importances)[0])
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects, rotation='vertical')
plt.ylabel('Importance')
plt.title('Feature Importances using Random forest')
plt.show()

# Feature Importance
# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(train.iloc[:,2:], train.iloc[:,1])


# Print the name and gini importance of each feature
feature_importances = sorted(zip(model.feature_importances_, list(train.columns)[2:]), reverse=True)
objects = (list(zip(*feature_importances)[1]))
y_pos = np.arange(len(objects))
performance = np.array(zip(*feature_importances)[0])
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects, rotation='vertical')
plt.ylabel('Importance')
plt.title('Feature Importances using Extra Trees Classifier')
plt.show()

X = train.iloc[:,2:]
y = train.iloc[:,1]
# fit model no training data
model = XGBClassifier()
model.fit(X, y)
# plot
feature_importances = sorted(zip(model.feature_importances_, list(train.columns)[2:]), reverse=True)
objects = (list(zip(*feature_importances)[1]))
y_pos = np.arange(len(objects))
performance = np.array(zip(*feature_importances)[0])
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects, rotation='vertical')
plt.ylabel('Importance')
plt.title('Feature Importances using XGBoost')
plt.show()




# Correction of Train Dat
# Correction Matrix Plot
names = train.columns
correlations = train.corr()
# plot correlation matrix
plt.rcParams['figure.figsize'] = (15,12)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,59,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names, rotation=90)
ax.set_yticklabels(names)
plt.show()






# Analysis Missing Data
train_missing_count = (train == -1).sum()
plt.rcParams['figure.figsize'] = (15,8)
train_missing_count.plot.bar()
plt.show()


test_missing_count = (test == -1).sum()
test_missing_count.plot.bar()
plt.show()

target = train_set['target'].copy()
train_set = train_set.drop(['id', 'target'], axis=1)

pd.value_counts(pd.value_counts(Train_set['ps_reg_03'].values['-1'], sort=False).values, sort=False)
train_set['ps_reg_03'].value_counts()
(train_set['ps_reg_03'] == -1).sum()
 
print("Feature Name    Unique     NaNCount")

for column in train_set:
    print("{0:15} {1:6d} {2:6}".format(column, Train_set[column].nunique(), (Train_set[column] == -1).sum()))

# Drop Columns that has -1 because the feature has missing from observations
for columns in train_set:
    if '-1' in columns:
        del train_set[columns]

# Train_set.drop(columns for columns in Train_set.columns if "-1" in columns], axis=1, inplace=True)

for col_names in train_set:
    if 'ps_calc' in col_names:
        del train_set[col_names]        
        
# Count the numbers of int64, float64, bool or object/string features
int_features = train_set.select_dtypes(include = ['int64']).columns.values
float_features = train_set.select_dtypes(include = ['float64']).columns.values
bool_features= train_set.select_dtypes(include = ['bool']).columns.values
categorical_features = train_set.select_dtypes(include = ['object']).columns.values
print('int_features:', int_features)
print('float_features:', float_features)
print('bool_features:', bool_features)
print('categorical_features:', categorical_features)



# Correlations Between Features
correlations = Train_set.corr(method='pearson')
print(correlations)

correlations = (Train_set.iloc[:,2:]).corr()
fig = plt.figure(figsize = (48, 24))
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0, n_col-2, 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(X.columns, rotation = 45)
ax.set_yticklabels(X.columns, rotation = 45)
plt.show()

# Skew for each features
skew = Train_set.skew()
print(skew)

# Visualization - Histograms
Train_set.hist()
plt.show()

Train_set[2:8].hist(figsize=(12,12))
plt.show()

# Visualization - Density Plots for distribution for each features
Train_set.plt(kind='density', subplots=True, layout=(3,3), sharex=False)
plt.show()

# Visualization - Scatterplots
scatter_matrix(Train_set)
plt.show()

# Visualization - Boxplots
Train_set.plt(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
plt.show()

# Visualization - Correlations Matrix Plot
Train_set.corr()
fig, ax = plt.subplots(figsize=(14,14))
ax.matshow(corr)
plt.xticks(range(len(corr.columns)), corr.columns)
plt.yticks(range(len(corr.columns)), corr.columns)


