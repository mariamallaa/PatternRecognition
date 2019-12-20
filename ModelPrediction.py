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
from skimage import io
from commonfunctions import *
import cv2
import csv
def predict(features):


    #features = sc.fit_transform(features)

    
    ohe = OneHotEncoder()
    # load json and create model
    json_file = open('model2.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model2.h5")
    
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
