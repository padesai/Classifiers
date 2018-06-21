## Data and Visual Analytics - Homework 4
## Georgia Institute of Technology
## Applying ML algorithms on the seizure dataset.

import numpy as np
import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import SVC

######################################### Reading and Splitting the Data ###############################################

# Read in all the data.
data = pd.read_csv('seizure_dataset.csv')

# Separate out the x_data and y_data.
x_data = data.loc[:, data.columns != "y"]
y_data = data.loc[:, "y"]

# The random state to use while splitting the data.
random_state = 100 # DO NOT CHANGE

# XXX
# TODO: Split 70% of the data into training and 30% into test sets. Call them x_train, x_test, y_train and y_test.
# Use the train_test_split method in sklearn with the parameter 'shuffle' set to true and the 'random_state' set to 100.
# XXX

x_train, x_test = train_test_split(x_data,shuffle=True,random_state = random_state,test_size=0.3)
y_train, y_test = train_test_split(y_data,shuffle=True,random_state = random_state,test_size=0.3)

# ###################################### Without Pre-Processing Data ##################################################
# XXX
# TODO: Fit the SVM Classifier (with the tuned parameters)  on the x_train and y_train data.
# XXX

svc_clf = SVC(C=0.0001,kernel='linear')

svc_clf.fit(x_train,y_train)

# XXX
# TODO: Predict the y values for x_test and report the test accuracy using the accuracy_score method.
# XXX

y_predict = svc_clf.predict(x_test)

print accuracy_score(y_test,y_predict)

# ###################################### With Data Pre-Processing ##################################################
# XXX
# TODO: Standardize or normalize x_train and x_test using either StandardScalar or normalize().
# Call the processed data x_train_p and x_test_p.
# XXX

x_train_p = normalize(x_train)
x_test_p = normalize(x_test)

# XXX
# TODO: Fit the SVM Classifier (with the tuned parameters) on the x_train_p and y_train data.
# XXX

svc_clf_normalized = SVC(C=0.0001,kernel='linear')

svc_clf_normalized.fit(x_train_p,y_train)

# XXX
# TODO: Predict the y values for x_test_p and report the test accuracy using the accuracy_score method.
# XXX


predict_y_normal = svc_clf_normalized.predict(x_test_p)

print accuracy_score(y_test, predict_y_normal)


