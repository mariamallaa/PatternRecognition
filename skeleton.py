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
separators = words_segmentation(corrected, lines_indices)
# drawing rectangles around words
words = binary.copy()
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
word = binary[
    # lines_indices[2] : lines_indices[3],separators[2][4] : separators[2][5]
    # lines_indices[7] : lines_indices[8],separators[7][2] : separators[7][3]
    # lines_indices[0] : lines_indices[1],separators[0][1] : separators[0][2],
    # lines_indices[5] : lines_indices[6],separators[5][7] : separators[5][8]
    # lines_indices[1] : lines_indices[2],separators[1][9] : separators[1][10],
    # lines_indices[1] : lines_indices[2],separators[1][12] : separators[1][13],
    # lines_indices[3] : lines_indices[4],separators[3][6] : separators[3][7],
    # lines_indices[1] : lines_indices[2],separators[1][8] : separators[1][9],
    # lines_indices[4] : lines_indices[5],separators[4][9] : separators[4][10],
    # lines_indices[0] : lines_indices[1],separators[0][8] : separators[0][9],
    # lines_indices[0] : lines_indices[1],separators[0][0] : separators[0][1],
    # lines_indices[1] : lines_indices[2],separators[1][3] : separators[1][4],
    # lines_indices[2] : lines_indices[3],separators[2][9] : separators[2][10],
    lines_indices[1] : lines_indices[2],
    separators[1][0] : separators[1][1],
    # lines_indices[2] : lines_indices[3],separators[2][4] : separators[2][5],
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


cutIndices = []
for i in range(separationIndices.shape[0] - 1):
    midRegion = separationIndices[i, 1] + int(
        (separationIndices[i + 1, 0] - separationIndices[i, 1]) / 2
    )
    while vp[midRegion] > mvf:
        midRegion += 1
    if len(cutIndices) == 0:
        cutIndices.append(midRegion)
    # 3ashan lamma el7oroof beyet2esem menha 7etta soghayara keda lwahdaha
    elif midRegion - cutIndices[-1] > 2:
        cutIndices.append(midRegion)

# strokes detection
cutIndices.insert(0, 0)
cutIndices.append(wordSkeleton.shape[1] - 1)
cutIndices = list(dict.fromkeys(cutIndices))
baselineIndex = np.argmax(np.sum(wordSkeleton, axis=1))

strokesIndices = []
length = len(cutIndices) - 1
for i in range(length):
    # to remove el7etat elfadya w el7etat elsoghayara el me2atta3a
    if np.sum(wordSkeleton[:, cutIndices[i] : cutIndices[i + 1]]) < 3:
        cutIndices.pop(i + 1)
        length -= 1
    if i + 1 >= length:
        break

    segment = wordSkeleton[:, cutIndices[i] : cutIndices[i + 1]]
    sumTopProjection = (np.sum(segment[0:baselineIndex, :], axis=1)).sum()
    sumBottomProjection = (np.sum(segment[baselineIndex + 1 :, :], axis=1)).sum()
    if sumTopProjection > sumBottomProjection:
        vp = np.sum(segment[:baselineIndex, :], axis=0)
        strokesHeight = np.max(vp)
        if strokesHeight < 0.25 * baselineIndex:
            hp = np.sum(segment[:baselineIndex, :], axis=1)
            hp = hp[hp != 0]

            if stats.mode(hp).mode[0] == mvf:

                strokesIndices.append(i)
    if i == length - 1:
        break

# print(len(strokesIndices))
# if len(strokesIndices) > 3:
#     for i in range(len(strokesIndices) - 2):
#         if (
#             strokesIndices[i + 2] - strokesIndices[i + 1] == 1
#             and strokesIndices[i + 1] - strokesIndices[i] == 1
#         ):
#             cutIndices.pop(strokesIndices[i + 2])
#             cutIndices.pop(strokesIndices[i + 1])

#             strokesIndices.pop(i + 2)
#             strokesIndices.pop(i + 1)


printWord = word.copy()
word[:, cutIndices] = 0.5
wordSkeleton[:, cutIndices] = 0.5
strokes = []
for i in range(len(strokesIndices)):
    strokes.append(cutIndices[strokesIndices[i]])
wordSkeleton[:, strokes] = 0.3


# viewing the images
viewer = CollectionViewer(
    [img, 1 - corrected, lines_segmented, words, printWord, word, wordSkeleton]
)
viewer.show()

