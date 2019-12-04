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
from skimage.morphology import thin

# reading the image
img = io.imread("scanned\capr2.png")

# skew correct with bounding rect
corrected = correct_skew(img)
corrected = np.round(corrected)

# indices of lines
lines_indices = line_segmentation(corrected)
# image with lines
lines_segmented = corrected.copy()
lines_segmented[lines_indices] = 0.5
# images into words
separators = words_segmentation(corrected, lines_indices)

words = corrected.copy()
for i in range(len(lines_indices) - 1):
    for j in range(len(separators[i]) - 1):
        words = cv2.rectangle(
            words,
            (separators[i][j], lines_indices[i]),
            (separators[i][j + 1], lines_indices[i + 1]),
            0.5,
            1,
        )

# word
first = corrected[
    lines_indices[0] : lines_indices[1], separators[0][1] : separators[0][2]
]

# character segmentation for a line
line = corrected[lines_indices[0] : lines_indices[1], :]
np.round(line)
projection = np.sum(line, axis=1)
baselineIndex = np.argmax(projection)
line[baselineIndex, :] = 0.5
verticalChange = []
for i in range(baselineIndex):
    verticalChange.append(len(np.where(line[i, :-1] != line[i, 1:])[0]))
verticalChange = np.asarray(verticalChange)
maxChangeIndex = np.argmax(verticalChange)

line[maxChangeIndex, :] = 0.3

# viewing the images
viewer = CollectionViewer([img, 1 - corrected, lines_segmented, words, line])
viewer.show()

