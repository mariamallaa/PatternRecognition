import tensorflow.keras
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
#Dependencies
import numpy as np
import pandas as pd
from skimage import io
from commonfunctions import *
import cv2
import csv





currentXlist=[]
currentylist=[]
f = open("Association\\training2.txt", "r")
k=1
for x in f:
    temp= x.split(" ",1) #maxsplit
    print(k)
    k+=1
    mycurrentchar=io.imread(temp[0])
    gray = cv2.cvtColor(mycurrentchar, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #resized28 = cv2.resize(binary, (28,28))
    #features = np.ravel(resized28)
    currentXlist.append(binary)
    y=temp[1].split("\n",1) #maxsplit
    currentylist.append(y[0])

currentX=np.asarray(currentXlist)
currenty=np.asarray(currentylist)
X_train,X_test,y_train,y_test = train_test_split(currentX,currenty,test_size = 0.4,random_state=109)
X_testing,X_validate,y_testing,y_validate = train_test_split(X_test,y_test,test_size = 0.5,random_state=109)
X_trainLast=np.zeros([X_train.shape[0],784])
X_testingLast=np.zeros([X_testing.shape[0],784])
X_validateLast=np.zeros([X_validate.shape[0],784])
# Normalize the images.
for i in range(len(X_train)):
    #X_train[i]= (X_train[i] / 255) - 0.5
    X_trainLast[i]= np.ravel(X_train[i])
    
for i in range(len(X_testing)):
    #X_testing[i] = (X_testing[i] / 255) - 0.5
    X_testingLast[i]= np.ravel(X_testing[i])
for i in range(len(X_validate)):
    #X_validate[i] = (X_validate[i] / 255) - 0.5
    X_validateLast[i]= np.ravel(X_validate[i])



model = Sequential()
print("1")
model.add(Dense(600, input_dim=784, activation='relu'))
model.add(Dense(130, activation='relu'))
model.add(Dense(29, activation='softmax'))

model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)


model.fit(
  X_trainLast,
  to_categorical(y_train),
  epochs=50,
  batch_size=64,
  validation_data=(X_validateLast, to_categorical(y_validate))
)

model.evaluate(
  X_testingLast,
  to_categorical(y_testing)
)


# serialize model to JSON
model_json = model.to_json()
with open("model5.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model5.h5")
print("Saved model to disk")


