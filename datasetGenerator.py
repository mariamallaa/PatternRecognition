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

# reading the image
# img = io.imread("scanned\capr2.png")

#

text_path = "C:\\Users\\Mariam Alaa\\Documents\\GitHub\\PatternRecognition\\text"

text_files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(text_path):
    for file in f:
        if ".txt" in file:
            text_files.append(os.path.join(r, file))


scanned_path = "C:\\Users\\Mariam Alaa\\Documents\\GitHub\\PatternRecognition\\scanned"
scanned_files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(scanned_path):
    for file in f:
        if ".png" in file:
            scanned_files.append(os.path.join(r, file))

for i in range(len(scanned_files)):
    print("img:", i)
    words = segment(scanned_files[i])
    cuts = np.asarray(words)
    cuts = cuts[:, 1]
    if segmentation_accuracy(text_files[i], cuts):
        labeling(i, words, text_files[i])
    else:
        print("failed at image: ", scanned_files[i])
        f = open("failed.txt", "a+")
        f.write(scanned_files[i] + "\n")
