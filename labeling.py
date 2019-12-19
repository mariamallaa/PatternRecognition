"""from enum import Enum, auto
import numpy as np
"""
import cv2

import numpy as np

import png

from PIL import Image

from skimage.viewer import ImageViewer

import matplotlib as pt
from skimage.transform import resize


def cutimg(word):
    charseg = []
    for cut in range(len(word[1]) - 1, 0, -1):

        char = word[0][:, word[1][cut - 1] : word[1][cut]]

        charseg.append(char)
    return charseg


def labeling(currentImg, words, path):
    f = open(path, encoding="utf-8")
    ################ 0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16   17  18  19  20  21  22   23  24  25  26  27
    characterlist = [
        "ا",
        "ب",
        "ت",
        "ث",
        "ج",
        "ح",
        "خ",
        "د",
        "ذ",
        "ر",
        "ز",
        "س",
        "ش",
        "ص",
        "ض",
        "ط",
        "ظ",
        "ع",
        "غ",
        "ف",
        "ق",
        "ك",
        "ل",
        "م",
        "ن",
        "ه",
        "و",
        "ي",
    ]
    contents = f.read()

    ArbSynFinal = []
    for bca in contents:  # Converting from unicode to arabic done
        encoded = bca.encode("utf-8")  # this works fine
        encoded = u"".join([c for c in bca])
        # print(encoded)
        ArbSynFinal.append(encoded)
    print("------Arb Syn Final----------")
    print(ArbSynFinal)
    classes = []
    lamalf = 0
    for fileletterindex in range(len(ArbSynFinal)):
        if ArbSynFinal[fileletterindex] == " " or ArbSynFinal[fileletterindex] == "\n":
            classes.append(-1)

        if lamalf == 1:
            lamalf = 0
            continue

        if ArbSynFinal[fileletterindex] == "ل":
            if ArbSynFinal[fileletterindex + 1] == "ا":
                classes.append(28)
                lamalf = 1
                continue

        for i in range(len(characterlist)):
            if ArbSynFinal[fileletterindex] == characterlist[i]:
                classes.append(i)
                break

    mycurrentindex = 0

    namefile = "training"
    f = open("Association\\" + namefile + ".txt", "a+")
    currentword = 1
    for word in words:
        charseg = cutimg(word)
        length = 0
        if mycurrentindex == 0:
            if classes[mycurrentindex] != -1:
                length += 1

        if mycurrentindex == len(classes) - 1:
            break
        while mycurrentindex != len(classes) - 1 and classes[mycurrentindex + 1] != -1:

            length += 1
            mycurrentindex += 1

        mycurrentindex += 1

        name = "segmentimage"
        index = 0
        print(len(charseg))
        if (len(charseg)) == length:

            currentindex = mycurrentindex - length

            for char in charseg:
                name = (
                    "img" + currentImg + name + str(currentword) + str(index) + ".png"
                )

                char = resize(char, (50, 50))
                # ret, bw_img = cv2.threshold(char,127,255,cv2.THRESH_BINARY)
                pt.image.imsave(name, char)

                f.write(name + " " + str(classes[currentindex]) + "\n")
                currentindex += 1

                index += 1
                name = "segmentimage"

        currentword += 1
    f.close()
