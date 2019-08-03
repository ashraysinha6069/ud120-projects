#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""
from __future__ import division
import pickle
import sys
import numpy as np
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list,sort_keys = '../tools/python2_lesson14_keys.pkl')
labels, features = targetFeatureSplit(data)

### Training and testing sets
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.30, random_state=42)

### Decision tree 
clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

### evaluation

print("accuracy:", accuracy_score(labels_test, pred))
values, counts = np.unique(pred, return_counts=True)
true_count=0
for i in range(0,len(labels_test)):
	if(pred[i]==1 and labels_test==1):
		true_count=true_count+1
print(true_count)

print("precision score:", precision_score(labels_test, pred))
print("recall score:", recall_score(labels_test, pred)) 
		
	
