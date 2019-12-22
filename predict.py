import math
from skimage.feature import greycomatrix, greycoprops
from skimage import io
from skimage.color import rgb2gray
from skimage.viewer import ImageViewer
from skimage.viewer import CollectionViewer
import matplotlib.pyplot as plt
import os
import numpy as np
from commonfunctions import *
from Segmentation import *
from labeling import *
import cv2
from skimage.morphology import thin, skeletonize
from scipy import stats
from datetime import datetime
from ModelPrediction import *

from SVMLoader import *
from Features import *
import time


print("hi")
# change it lel directory beta3 input

# load json and create model
json_file = open('model5.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model5.h5")


    
# evaluate loaded model on test data
loaded_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


scanned_path = "C:\\Users\\Mariam Alaa\\Documents\\GitHub\\new\\PatternRecognition\\scanned"
scanned_files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(scanned_path):
    for file in f:
        if ".png" in file:
            scanned_files.append(os.path.join(r, file))

characterlist = [
        "ا",
        "ب",
        "ت",
        "ث",
        "ج",
        "ح",
        "خ",
        "د",
        "ذ",
        "ر",
        "ز",
        "س",
        "ش",
        "ص",
        "ض",
        "ط",
        "ظ",
        "ع",
        "غ",
        "ف",
        "ق",
        "ك",
        "ل",
        "م",
        "ن",
        "ه",
        "و",
        "ي",
        "لا"
    ]
running_time = open("output\\running_time.txt", "a+")
for i in range(len(scanned_files)):
    print("img:", i)
    #start = datetime.now()
    start_time = time.time()
    # reading the image and segmenting into words and characters
    words = segment(scanned_files[i])
    words=np.asarray(words)
    index=1
    f = open("output\\test\\please_" + str(i + 1) + ".txt", "a+", encoding="utf-8")
    #f = open("output\\test\\test_" + str(i + 1) + ".txt", "a+", encoding="utf-8")
    generatedWords=[]
    for j in range(len(words)):
        generatedWord = ""
        word = words[j, 0]
        
        cut_indices = words[j, 1]
        for k in range(len(cut_indices)-1,0,-1):
            letter = word[:, cut_indices[k - 1] : cut_indices[k]]
           
            resized28 = cv2.resize(letter, (28, 28),interpolation = cv2.INTER_NEAREST)


            features = np.ravel(resized28.astype('uint8'))

            #features=Combine(letter.astype('uint8'))

            

    

            y_pred = loaded_model.predict([[features]])
            letter_class=[]
            for i in range(len(y_pred)):
                letter_class.append(np.argmax(y_pred[i]))




            #letter_class = PredictSVM([features])
            
            generatedWord+=characterlist[letter_class[0]]
            
        #f.write(generatedWord+" ")
        generatedWords.append(generatedWord+" ")
        print(index)
        index+=1
        generatedWord=""

    #end=datetime.now()
    running_time.write(str(time.time() - start_time)+"\n")
    
    for ind in range(len(generatedWords)):
        f.write(generatedWords[ind])
    f.close()
    









