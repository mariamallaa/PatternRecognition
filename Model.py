

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
#Dependencies
import numpy as np
import pandas as pd
from skimage import io
from commonfunctions import *
import cv2
import csv

file2 = open(r"D:\\pattern dataset\\Xtest.txt","w+")
file3 = open(r"D:\\pattern dataset\\Ytest.txt","w+")

def get_xy():
    currentX=[]
    currenty=[]
    img=np.zeros([28,28])
    f = open("C:\\Users\\Mariam Alaa\\Documents\\GitHub\\PatternRecognition\\Association\\training.txt", "r")
    for x in f:
        temp= x.split(" ",1) #maxsplit
        
        mycurrentchar=io.imread(temp[0])
        gray = cv2.cvtColor(mycurrentchar, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        resized28 = cv2.resize(binary, (28,28))
        features = np.ravel(resized28)
        currentX.append(features)
        y=temp[1].split("\n",1) #maxsplit
        currenty.append(y[0])
        
   
    
    
    return [currentX,currenty]
        

    f.close()
        

X,Y=get_xy()


for features in X:
    for _string in features:
        file2.write(str(_string))
    file2.write('\n')
file2.close()
for labels in Y:
    for _string in labels:
        file3.write(str(_string))
    file3.write('\n')
file3.close()


'''
y=[]
X=[]
filepath = "D:\\pattern dataset\\Y.txt"
with open(filepath) as fp:
   line = fp.readline()
   y.append([int(line)])
   print(y)
   while line:
       line = fp.readline()
       #print(line)
       if line!='':
            y.append([int(line)])
       

filepath = "D:\\pattern dataset\\X.txt"
with open(filepath) as fp:
   line = fp.readline()
   while line:
        array=[]
        for char in line:
            if char!='\n':
                array.append(int(char))
        X.append(array)
        line = fp.readline()
        #X.append(line)


#Normalizing the data



# Neural network
model = Sequential()
print("1")
model.add(Dense(16, input_dim=784, activation='relu'))
model.add(Dense(120, activation='relu'))
model.add(Dense(29, activation='softmax'))
print("3")
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.4,random_state=109)
X_testing,X_validate,y_testing,y_validate = train_test_split(X_test,y_test,test_size = 0.5,random_state=109)
print("4")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print("5")
history = model.fit(X_train, y_train,validation_data = (X_validate,y_validate), epochs=100, batch_size=64)
print("6")



y_pred = model.predict(X_testing)
#Converting predictions to label
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
#Converting one hot encoded test label to label
test = list()
for i in range(len(y_testing)):
    test.append(np.argmax(y_testing[i]))

a = accuracy_score(pred,test)
print('Accuracy is:', a*100)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss']) 
plt.plot(history.history['val_loss']) 
plt.title('Model loss') 
plt.ylabel('Loss') 
plt.xlabel('Epoch') 
plt.legend(['Train', 'Test'], loc='upper left') 
plt.show()

 
# serialize model to JSON
model_json = model.to_json()
with open("model2.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model2.h5")
print("Saved model to disk")
'''