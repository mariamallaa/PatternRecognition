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
from test import *
from labeling import *
import cv2
from skimage.morphology import thin, skeletonize
from scipy import stats

# reading the image
# img = io.imread("scanned\capr2.png")

#
def segmentation_accuracy(path, words):
    text = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            for word in line.split():
                length = len(word)
                for i in range(len(word) - 1):
                    if word[i] == "ل" and word[i + 1] == "ا":

                        length -= 1
                text.append(length)

    if len(text) == len(words):

        correct = 0
        for i in range(len(words)):
            if len(words[i]) - 1 == text[i]:
                correct += 1
        print("character segmentation accuracy=", (correct / len(words)) * 100)
        return True
    else:
        print("word segmentation failed", len(text), len(words))
        return False


def segmentation(path):
    img = io.imread(path)

    # skew correct with bounding rect
    corrected = correct_skew(img)
    ret2, th2 = cv2.threshold(
        corrected, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    binary = th2 / 255
    corrected = 1 - (corrected / 255)

    # indices of lines
    lines_indices = line_segmentation(corrected)
    # images into words
    separators = words_segmentation(binary, lines_indices)
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

            strokes, cutIndices = character_segmentation(
                word, wordSkeleton, baselineIndex, maxChangeIndex, topIndex, bottomIndex
            )
            words.append([word, cutIndices])
    return words

    labeling(words)


text_path = "C:\\Users\\Maram\\Downloads\\text"

text_files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(text_path):
    for file in f:
        if ".txt" in file:
            text_files.append(os.path.join(r, file))


scanned_path = "C:\\Users\\Maram\\Downloads\\scanned"
scanned_files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(scanned_path):
    for file in f:
        if ".png" in file:
            scanned_files.append(os.path.join(r, file))

for i in range(len(scanned_files)):
    print("img:", i)
    words = segmentation(scanned_files[i])
    cuts = np.asarray(words)
    cuts = cuts[:, 1]
    if segmentation_accuracy(text_files[i], cuts):
        labeling(i, words, text_files[i])
    else:
        print("failed at image: ", scanned_files[i])
        f = open("failed.txt", "a+")
        f.write(scanned_files[i] + "\n")
