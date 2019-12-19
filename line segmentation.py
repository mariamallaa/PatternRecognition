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
from accuracy import *
from test2 import *

# from test import *
import cv2
from skimage.morphology import thin, skeletonize
from scipy import stats

# reading the image
# img = io.imread("scanned\capr2.png")

#
img = io.imread("scanned\csep1638.png")

# skew correct with bounding rect
corrected = correct_skew(img)
# blur = cv2.GaussianBlur(corrected, (3, 3), 0)
ret2, th2 = cv2.threshold(corrected, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
binary = th2 / 255
corrected = 1 - (corrected / 255)

# indices of lines
lines_indices = line_segmentation(corrected)
# image with lines
lines_segmented = corrected.copy()
lines_segmented[lines_indices] = 0.5


# images into words
separators = words_segmentation(binary, lines_indices)
# drawing rectangles around words
wordsRects = binary.copy()
for i in range(len(lines_indices) - 1):
    for j in range(len(separators[i]) - 1):
        wordsRects = cv2.rectangle(
            wordsRects,
            (separators[i][j], lines_indices[i]),
            (separators[i][j + 1], lines_indices[i + 1]),
            0.5,
            1,
        )

# view = ImageViewer(wordsRects)
# view.show()

words = []
for i in range(len(lines_indices) - 1):
    # finding baseline index for the entire line
    line = binary[lines_indices[i] : lines_indices[i + 1]]
    # line = skeletonize(line).astype(np.float)
    projection = np.sum(line, axis=1)
    # line = line[projection != 0]
    baselineIndex = np.argmax(projection)
    # getting start of line
    topIndex = 0
    while topIndex < len(projection):
        if projection[topIndex] == 0:
            topIndex += 1
        else:
            break
    print("Top index=", topIndex)
    bottomIndex = len(projection) - 1
    # getting end of line
    while bottomIndex > 0:
        if projection[topIndex] == 0:
            topIndex -= 1
        else:
            break
    print("Bottom index=", bottomIndex)
    verticalChange = []
    for k in range(baselineIndex):
        verticalChange.append(len(np.where(line[k, :-1] != line[k, 1:])[0]))
    verticalChange = np.asarray(verticalChange)
    maxChangeIndex = np.argmax(verticalChange)

    for j in range(len(separators[i]) - 1, 0, -1):
        # separating words using indices
        word = line[
            :, separators[i][j - 1] : separators[i][j],
        ]
        wordSkeleton = skeletonize(word).astype(np.float)
        # character segmentation for a word
        # wordSkeleton = skeletonize(word).astype(np.float)
        strokes, cutIndices = character_segmentation(
            word, wordSkeleton, baselineIndex, maxChangeIndex, topIndex, bottomIndex
        )
        words.append([wordSkeleton, cutIndices])
        # cut = wordSkeleton.copy()
        # cut[:, strokes] = 0.3
        # wordSkeleton[:, strokes] = 0.3
        # wordSkeleton[:, cutIndices] = 0.5
        # view = CollectionViewer([wordSkeleton, cut])
        # view.show()


words = np.asarray(words)
# segmentation_accuracy("text\capr2.txt", words[:, 1])

segmentation_accuracy("text\csep1638.txt", words[:, 1])
# printWord = word.copy()
# word[:, cutIndices] = 0.5
# wordSkeleton[:, cutIndices] = 0.5
# strokes = []
# for i in range(len(strokesIndices)):
#     strokes.append(cutIndices[strokesIndices[i]])
# wordSkeleton[:, strokes] = 0.3


# viewing the images
viewer = CollectionViewer([img, 1 - corrected, lines_segmented, wordsRects])
viewer.show()

