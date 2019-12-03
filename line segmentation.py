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
from preprocessing import *
import cv2

#reading the image
img=io.imread("scanned\capr6.png")

#skew correct with bounding rect
corrected=correct_skew(img)
#indices of lines
lines_indices=line_segmentation(corrected)
#image with lines
lines_segmented=corrected.copy()
lines_segmented[lines_indices]=0.5
#images into words
# separators=words_segmentation(corrected,lines_indices)
# separated_words=lines_segmented.copy()
# separated_words[:,separators]=0.5

separators=words_segmentation(corrected,lines_indices)
print(separators[0][1])
words=corrected.copy()
for i in range(len(lines_indices)-1):
    for j in range(len(separators[i])-1):
        words= cv2.rectangle(words,(separators[i][j],lines_indices[i]),(separators[i][j+1],lines_indices[i+1]),0.5,1)

vertical=corrected.copy()
vertical[:,separators[0]]=0.5

first=corrected[lines_indices[0]:lines_indices[1],separators[0][1]:separators[0][2]]
#viewing the images
viewer=CollectionViewer([img, 1-corrected, lines_segmented,vertical,words])
viewer.show()


