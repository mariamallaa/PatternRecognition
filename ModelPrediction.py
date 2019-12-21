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
def predict(features,loaded_model):


    #features = sc.fit_transform(features)

    
    ohe = OneHotEncoder()
    
    
    # evaluate loaded model on test data
    loaded_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    y_pred = loaded_model.predict([features])
    pred=[]
    for i in range(len(y_pred)):
        pred.append(np.argmax(y_pred[i]))
    return pred
'''
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
'''
