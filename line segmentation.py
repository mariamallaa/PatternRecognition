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
from scipy import stats

# reading the image
img = io.imread("scanned\capr6.png")

# skew correct with bounding rect
corrected = correct_skew(img)

# indices of lines
lines_indices = line_segmentation(corrected)
# image with lines
lines_segmented = corrected.copy()
lines_segmented[lines_indices] = 0.5
# images into words
separators = words_segmentation(corrected, lines_indices)
# drawing rectangles around words
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

# converting to binary image
corrected = np.round(corrected)
# separating words using indices
first = corrected[
    #lines_indices[2] : lines_indices[3],separators[2][4] : separators[2][5]
    # lines_indices[7] : lines_indices[8],separators[7][2] : separators[7][3]
    # lines_indices[0] : lines_indices[1],separators[0][1] : separators[0][2],
    # lines_indices[5] : lines_indices[6],separators[5][7] : separators[5][8]
    # lines_indices[1] : lines_indices[2],separators[1][9] : separators[1][10],
    lines_indices[1] : lines_indices[2],separators[1][12] : separators[1][13],
]

# character segmentation for a word
# finding baseline index
projection = np.sum(first, axis=1)
baselineIndex = np.argmax(projection)

# finding maximum transition index
verticalChange = []
for i in range(baselineIndex):
    # print(np.where(first[i, :-1] != first[i, 1:])[0])
    verticalChange.append(len(np.where(first[i, :-1] != first[i, 1:])[0]))
verticalChange = np.asarray(verticalChange)
maxChangeIndex = max(np.argmax(verticalChange), baselineIndex - 3)
print(baselineIndex, maxChangeIndex)

# getting separation region indices
kernel = np.ones((2, 2), np.uint8)
first = cv2.morphologyEx(first, cv2.MORPH_CLOSE, kernel)
separationIndices = np.where(first[maxChangeIndex, :-1] != first[maxChangeIndex, 1:])[0]
separationIndices = separationIndices.reshape(-1, 2)
# getting cut indices

vp = np.sum(first, axis=0)
mvf = max(stats.mode(vp).mode[0], 2)
print(mvf)

cutIndices = []
for i in range(separationIndices.shape[0] - 1):
    midRegion = separationIndices[i, 1] + int(
        (separationIndices[i + 1, 0] - separationIndices[i, 1]) / 2
    )
    while vp[midRegion] > mvf:
        midRegion += 1
    if len(cutIndices) == 0:
        cutIndices.append(midRegion)
    elif midRegion - cutIndices[-1] != 1:
        cutIndices.append(midRegion)
word = first.copy()
first[:, cutIndices] = 0.5
# first[baselineIndex, :] = 0.5
# first[maxChangeIndex, :] = 0.3


# viewing the images
viewer = CollectionViewer([img, 1 - corrected, lines_segmented, words, word, first])
viewer.show()

