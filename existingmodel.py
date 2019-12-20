import tensorflow.keras
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import model_from_json
#Dependencies
import numpy as np
import pandas as pd
from skimage import io
from commonfunctions import *
import cv2
import csv

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

sc = StandardScaler()
X = sc.fit_transform(X)
ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


y_pred = loaded_model.predict(X)
#Converting predictions to label
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
#Converting one hot encoded test label to label
test = list()
for i in range(len(y)):
    test.append(np.argmax(y[i]))

a = accuracy_score(pred,test)
print('Accuracy is:', a*100)