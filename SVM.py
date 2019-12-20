import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
#from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
import pickle


y=[]
X=[]
filepath = "D:\\pattern dataset\\Y.txt"
with open(filepath) as fp:
   line = fp.readline()
   
   while line:
       if line!='':
            y.append(int(line))
       line = fp.readline()
       

filepath = "D:\\pattern dataset\\X.txt"
with open(filepath) as fp:
   line = fp.readline()
   while line:
        array=[]
        for char in line:
            if char!='\n':
                array.append(int(char))
        X.append(array)
        #print(array)
        line = fp.readline()
#cancer = datasets.load_breast_cancer()
print("finish reading x,y")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
#X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3,random_state=109)
svclassifier = SVC(kernel='rbf')
svclassifier.fit(X_train, y_train)
print("finish fitting")


#predict
y_pred = svclassifier.predict(X_test)

#print stats
'''
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("Precision:",metrics.precision_score(y_test, y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)
print("Recall:",metrics.recall_score(y_test, y_pred))
'''

filename = "finalized_model.sav"
f= open(filename,"w+")
pickle.dump(svclassifier, open(filename, 'wb'))
print("finish saving")