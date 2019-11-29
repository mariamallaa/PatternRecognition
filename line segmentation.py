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

img=io.imread("scanned\capr6.png")


corrected=adjust_tilt(img)

lines_indices=line_segmentation(corrected)

lines_segmented=corrected.copy()
lines_segmented[lines_indices]=0.5

viewer=CollectionViewer([corrected,lines_segmented])
viewer.show()
#show_images([img,corrected,divided])

