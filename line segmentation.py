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
img = io.imread("scanned\capr2.png")
# img = io.imread("scanned\csep1635.png")

# skew correct with bounding rect
corrected = correct_skew(img)

# indices of lines
lines_indices = line_segmentation(corrected)
# image with lines
lines_segmented = corrected.copy()
lines_segmented[lines_indices] = 0.5
# converting to binary image
binary = np.round(corrected)
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


# separating words using indices
first = binary[
    # lines_indices[2] : lines_indices[3],separators[2][4] : separators[2][5]
    # lines_indices[7] : lines_indices[8],separators[7][2] : separators[7][3]
    lines_indices[0] : lines_indices[1],
    separators[0][1] : separators[0][2],
    # lines_indices[5] : lines_indices[6],separators[5][7] : separators[5][8]
    # lines_indices[1] : lines_indices[2],separators[1][9] : separators[1][10],
    # lines_indices[1] : lines_indices[2],separators[1][12] : separators[1][13],
    # lines_indices[3] : lines_indices[4],separators[3][6] : separators[3][7],
    # lines_indices[1] : lines_indices[2],separators[1][8] : separators[1][9],
    # lines_indices[4] : lines_indices[5], separators[4][9] : separators[4][10],
    # lines_indices[0] : lines_indices[1],separators[0][8] : separators[0][9],
    # lines_indices[0] : lines_indices[1],separators[0][0] : separators[0][1],
    # lines_indices[1] : lines_indices[2],separators[1][4] : separators[1][5],
    # lines_indices[2] : lines_indices[3],separators[2][9] : separators[2][10],
]

# character segmentation for a word
kernel = np.ones((2, 2), np.uint8)
closed = cv2.morphologyEx(first, cv2.MORPH_CLOSE, kernel)

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

separationIndices = np.where(closed[maxChangeIndex, :-1] != closed[maxChangeIndex, 1:])[
    0
]
separationIndices = separationIndices.reshape(-1, 2)
# getting cut indices

vp = np.sum(closed, axis=0)
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

# strokes detection
cutIndices.insert(0, 0)
cutIndices.append(closed.shape[1] - 1)
cutIndices = list(dict.fromkeys(cutIndices))
closedBaselineIndex = np.argmax(np.sum(closed, axis=1))
# closedBaselineIndex = 23
print(closedBaselineIndex)
# print(cutIndices)
strokesIndices = []
for i in range(len(cutIndices) - 1):
    segment = closed[:, cutIndices[i] : cutIndices[i + 1]]
    sumTopProjection = (np.sum(segment[0:closedBaselineIndex, :], axis=1)).sum()
    sumBottomProjection = (np.sum(segment[closedBaselineIndex + 1 :, :], axis=1)).sum()
    # print(segment)
    # print(sumTopProjection, sumBottomProjection)
    if sumTopProjection > sumBottomProjection:
        vp = np.sum(segment[:closedBaselineIndex, :], axis=0)
        # print(vp)
        strokesHeight = np.max(vp)
        # print(strokesHeight)
        if strokesHeight < 0.25 * closedBaselineIndex:
            hp = np.sum(segment[:closedBaselineIndex, :], axis=1)
            hp = hp[hp != 0]

            # print(segment[:closedBaselineIndex, :])
            print(cutIndices[i], "mode", stats.mode(hp).mode[0])
            if stats.mode(hp).mode[0] == mvf:
                print(stats.mode(vp).mode[0], mvf)
                strokesIndices.append(cutIndices[i])


word = first.copy()
first[:, cutIndices] = 0.5
closed[:, cutIndices] = 0.5
closed[:, strokesIndices] = 0.3
print(strokesIndices)

# viewing the images
viewer = CollectionViewer(
    [img, 1 - corrected, lines_segmented, words, word, first, closed]
)
viewer.show()

