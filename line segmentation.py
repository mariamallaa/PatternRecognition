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
#viewing the images
viewer=CollectionViewer([img, 1-corrected, lines_segmented])
viewer.show()


