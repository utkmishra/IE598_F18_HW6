# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 10:04:28 2018

@author: Utkarsh Mishra
"""
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import time

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

In_Sample_Score=[]
Out_of_Sample_Score=[]

start = time.clock()

for i in range(1,11):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i, stratify=y)
    tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
    tree.fit(X_train, y_train)
    y_test_pred = tree.predict(X_test)
    y_train_pred = tree.predict(X_train)    
    In_score = metrics.accuracy_score(y_train, y_train_pred)
    Out_score = metrics.accuracy_score(y_test, y_test_pred)
    In_Sample_Score.append(In_score)    
    Out_of_Sample_Score.append(Out_score)
    print('Random State: %d, In-sample score: %.3f, Out of sample score: %.3f' % (i,In_score,Out_score))

print("In_Sample mean:",np.mean(In_Sample_Score))
print("In_Sample stddev:",np.std(In_Sample_Score))
print("Out_of_Sample mean:",np.mean(Out_of_Sample_Score))
print("Out_of_Sample stddev:",np.std(Out_of_Sample_Score))

stop = time.clock()

print ("Time Taken: %.2f sec" % (stop-start))


#Cross validation

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1, stratify=y)

start = time.clock()

kfold = KFold(n_splits=10,random_state=1).split(X_train, y_train)

tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)

tree.fit(X_train, y_train)

Scores = cross_val_score(estimator=tree, X=X_train, y=y_train, cv=10, n_jobs=1)

y_test_pred = tree.predict(X_test)
Out_of_Sample_Score=metrics.accuracy_score(y_test,y_test_pred)

print('CV accuracy scores: %s' % Scores)
print('CV accuracy mean/std: %.3f +/- %.3f' % (np.mean(Scores), np.std(Scores)))
print("Out of Sample Score:", Out_of_Sample_Score,'\n')
 
stop = time.clock()
   
print ("Time Taken: %.2f sec" % (stop-start))



print("My name is Utkarsh Mishra")
print("My NetID is: umishra3")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")