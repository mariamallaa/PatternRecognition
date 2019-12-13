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
from skimage.morphology import thin, skeletonize
from scipy import stats

# reading the image
img = io.imread("scanned\capr2.png")

#
# img = io.imread("scanned\csep1635.png")

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


# separating words using indices
word = binary[
    lines_indices[0] : lines_indices[1], separators[0][1] : separators[0][2],
]

# character segmentation for a word

wordSkeleton = skeletonize(word).astype(np.float)

# finding baseline index
projection = np.sum(wordSkeleton, axis=1)
baselineIndex = np.argmax(projection)

# finding maximum transition index
verticalChange = []
for i in range(baselineIndex):
    verticalChange.append(len(np.where(wordSkeleton[i, :-1] != wordSkeleton[i, 1:])[0]))
verticalChange = np.asarray(verticalChange)
maxChangeIndex = max(np.argmax(verticalChange), baselineIndex - 3)


# getting separation region indices

separationIndices = np.where(
    wordSkeleton[maxChangeIndex, :-1] != wordSkeleton[maxChangeIndex, 1:]
)[0]
separationIndices = separationIndices.reshape(-1, 2)

# getting cut indices
vp = np.sum(wordSkeleton, axis=0)
mvf = stats.mode(vp[vp != 0]).mode[0]
vpAbove = np.sum(wordSkeleton[:baselineIndex, :])
vpBelow = np.sum(wordSkeleton[baselineIndex + 1 :, :])
cutIndices = []
cutIndices.append(0)
for i in range(separationIndices.shape[0] - 1):
    midRegion = separationIndices[i, 1] + int(
        (separationIndices[i + 1, 0] - separationIndices[i, 1]) / 2
    )
    while vp[midRegion] > mvf:
        midRegion += 1

    # 3ashan lamma el7oroof beyet2esem menha 7etta soghayara keda lwahdaha
    if np.sum(wordSkeleton[:, cutIndices[-1] : midRegion]) > 3:
        if midRegion - cutIndices[-1] > 2:
            cutIndices.append(midRegion)

#
if np.sum(wordSkeleton[:, cutIndices[-1] : wordSkeleton.shape[1]]) > 3:
    cutIndices.append(wordSkeleton.shape[1] - 1)

cutIndices = list(dict.fromkeys(cutIndices))
# strokes detection

# baselineIndex = np.argmax(np.sum(wordSkeleton, axis=1))
# using line to get line index not word
# baselineIndex = np.argmax(
#     np.sum(binary[lines_indices[0] : lines_indices[1], :], axis=1)
# )
print(baselineIndex)
strokesIndices = []
length = len(cutIndices) - 1

for i in range(length):
    segment = wordSkeleton[:, cutIndices[i] : cutIndices[i + 1]]
    sumTopProjection = (np.sum(segment[0:baselineIndex, :], axis=1)).sum()
    sumBottomProjection = (np.sum(segment[baselineIndex + 1 :, :], axis=1)).sum()

    if sumTopProjection > sumBottomProjection:
        vp = np.sum(segment[:baselineIndex, :], axis=0)
        strokesHeight = np.max(vp)
        print(cutIndices[i], "passed condition1")
        h = np.sort(np.sum(segment, axis=1))[::-1][0]

        if strokesHeight < h:
            print(cutIndices[i], "passed condition2 strokes Height=", strokesHeight)
            hp = np.sum(segment[:baselineIndex, :], axis=1)
            hp = hp[hp != 0]

            if stats.mode(hp).mode[0] == mvf:
                print(cutIndices[i], "passed condition3")
                strokesIndices.append(i)
        elif len(strokesIndices) >= 2:
            if i - strokesIndices[-1] == 1 and i - strokesIndices[-2] == 2:
                if strokesHeight < 2 * h:
                    print(cutIndices[i], "passed condition4")
                    hp = np.sum(segment[:baselineIndex, :], axis=1)
                    hp = hp[hp != 0]

                    if stats.mode(hp).mode[0] == mvf:
                        print(cutIndices[i], "passed condition5")
                        strokesIndices.append(i)

# check that last letter is not split

lastSegment = wordSkeleton[:, cutIndices[0] : cutIndices[1]]
if (np.sum(lastSegment[baselineIndex + 1 :, :], axis=1)).sum() > (
    np.sum(lastSegment[0:baselineIndex, :], axis=1)
).sum():

    cutIndices.pop(1)

# make seen one letter instead of 3

# print(len(strokesIndices))
# if len(strokesIndices) > 2:
#     for i in range(len(strokesIndices) - 2):
#         print(i)
#         if (
#             strokesIndices[i + 2] - strokesIndices[i + 1] == 1
#             and strokesIndices[i + 1] - strokesIndices[i] == 1
#         ):
#             cutIndices.pop(strokesIndices[i + 2])
#             cutIndices.pop(strokesIndices[i + 1])

#             i += 2


printWord = word.copy()
word[:, cutIndices] = 0.5
wordSkeleton[:, cutIndices] = 0.5
strokes = []
for i in range(len(strokesIndices)):
    strokes.append(cutIndices[strokesIndices[i]])
wordSkeleton[:, strokes] = 0.3


# viewing the images
viewer = CollectionViewer(
    [img, 1 - corrected, lines_segmented, wordsRects, printWord, word, wordSkeleton]
)
viewer.show()

