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
filepath = "D:\\pattern dataset\\Ytest.txt"
with open(filepath) as fp:
   line = fp.readline()
   y.append([int(line)])
   print(y)
   while line:
       line = fp.readline()
       #print(line)
       if line!='':
            y.append([int(line)])
       

filepath = "D:\\pattern dataset\\Xtest.txt"
with open(filepath) as fp:
   line = fp.readline()
   while line:
        array=[]
        for char in line:
            if char!='\n':
                array.append(int(char))
        X.append(array)
        line = fp.readline()

print("have the data")



def PredictSVM(Features):
    filename = "finalized_model.sav"
    loaded_model = pickle.load(open(filename, 'rb'))
    #predict
    y_pred = loaded_model.predict(Features)
    print(y_pred)
    return y_pred