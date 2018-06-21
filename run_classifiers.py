## Data and Visual Analytics - Homework 4
## Georgia Institute of Technology
## Applying ML algorithms to recognize seizure from EEG brain wave signals

import numpy as np
import pandas as pd
import time

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

######################################### Reading and Splitting the Data ###############################################

# Read in all the data.
data = pd.read_csv('seizure_dataset.csv')

# Separate out the x_data and y_data.
x_data = data.loc[:, data.columns != "y"]
y_data = data.loc[:, "y"]

# The random state to use while splitting the data. DO NOT CHANGE.
random_state = 100 # DO NOT CHANGE

# XXX
# TODO: Split each of the features and labels arrays into 70% training set and
#       30% testing set (create 4 new arrays). Call them x_train, x_test, y_train and y_test.
#       Use the train_test_split method in sklearn with the parameter 'shuffle' set to true
#       and the 'random_state' set to 100.
# XXX

x_train, x_test = train_test_split(x_data,shuffle=True,random_state = random_state,test_size=0.3)
y_train, y_test = train_test_split(y_data,shuffle=True,random_state = random_state,test_size=0.3)
# ############################################### Linear Regression ###################################################
# XXX
# TODO: Create a LinearRegression classifier and train it.
# XXX
lin_reg = LinearRegression()
lin_reg.fit(x_train,y_train)

# XXX
# TODO: Test its accuracy (on the testing set) using the accuracy_score method.
# Note: Use y_predict.round() to get 1 or 0 as the output.
# XXX

y_predict = lin_reg.predict(x_test)
accuracy_score(y_test,y_predict.round())

print accuracy_score(y_test,y_predict.round())
y_train_predict = lin_reg.predict(x_train)
print accuracy_score(y_train,y_train_predict.round())


# ############################################### Multi Layer Perceptron #################################################
# XXX
# TODO: Create an MLPClassifier and train it.
# XXX
clf = MLPClassifier()
clf.fit(x_train,y_train)

# XXX
# TODO: Test its accuracy on the test set using the accuracy_score method.
# XXX

y_predict_mlp = clf.predict(x_test)

accuracy_score(y_test,y_predict_mlp)

print accuracy_score(y_test,y_predict_mlp)
y_train_predict_mlp = clf.predict(x_train)
print accuracy_score(y_train,y_train_predict_mlp)

# ############################################### Random Forest Classifier ##############################################
# XXX
# TODO: Create a RandomForestClassifier and train it.
# XXX

clf_randomforest = RandomForestClassifier()
clf_randomforest.fit(x_train,y_train)

# XXX
# TODO: Test its accuracy on the test set using the accuracy_score method.
# XXX

y_predict_rf = clf_randomforest.predict(x_test)
accuracy_score(y_test,y_predict_rf)

print accuracy_score(y_test,y_predict_rf)
y_train_predict_rf = clf_randomforest.predict(x_train)
print accuracy_score(y_train,y_train_predict_rf)

# XXX
# TODO: Tune the hyper-parameters 'n_estimators' and 'max_depth'.
#       Print the best params, using .best_params_, and print the best score, using .best_score_.
# XXX

rfc = RandomForestClassifier()
param_grid = {
    'n_estimators': [20,30,40,50,60],
    'max_depth': [4,5,6,7,8,9,10]
}

cv_rfc = GridSearchCV(estimator=rfc,param_grid=param_grid, cv = 10)

cv_rfc.fit(x_train,y_train)

print cv_rfc.best_params_
print cv_rfc.best_score_



# ############################################ Support Vector Machine ###################################################
# XXX
# TODO: Create a SVC classifier and train it.
# XXX

clf_svc = SVC()
clf_svc.fit(x_train,y_train)

# XXX
# TODO: Test its accuracy on the test set using the accuracy_score method.
# XXX

y_predict_svc = clf_svc.predict(x_test)
accuracy_score(y_test,y_predict_svc)

print accuracy_score(y_test,y_predict_svc)
y_train_predict_svc = clf_svc.predict(x_train)
print accuracy_score(y_train,y_train_predict_svc)



# XXX
# TODO: Tune the hyper-parameters 'C' and 'kernel' (use rbf and linear).
#       Print the best params, using .best_params_, and print the best score, using .best_score_.
# XXX

support_vector_classifieer = SVC()

Cs = [0.0001,0.001,0.01,0.1,1.0,10.0,100.0,1000.0]
kernels = ["rbf","linear"]
param_grid = {'C': Cs, 'kernel' : kernels}
cv_svc = GridSearchCV(estimator=support_vector_classifieer,param_grid=param_grid, cv=10)

x_train_p = normalize(x_train)
x_test_p = normalize(x_test)
cv_svc.fit(x_train_p,y_train)

print cv_svc.best_params_
print cv_svc.best_score_


# XXX
# ########## PART C #########
# TODO: Print your CV's highest mean testing accuracy and its corresponding mean training accuracy and mean fit time.
# 		State them in report.txt.
# XXX

print cv_svc.cv_results_['mean_test_score']
print cv_svc.cv_results_['mean_train_score']
print cv_svc.cv_results_['mean_fit_time']
