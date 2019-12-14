import numpy as np
from scipy.ndimage import interpolation as inter
from scipy import stats
import cv2
from skimage.viewer import ImageViewer


def correct_skew(img):
    thresh = img.copy()
    thresh = 1 - (thresh / 255)

    coords = np.column_stack(np.where(thresh > 0))

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    # rotated = 1 - (rotated / 255)
    # rotated = np.round(rotated)
    return rotated


def line_segmentation(img):
    projection = np.sum(img, axis=1)

    lines_indices = []
    i = 0
    start = 0
    end = 0
    while i < len(projection):
        if projection[i] == 0:
            start = i
            j = 0

            while projection[j + i] == 0 and j + i < len(projection):
                j += 1
                if i + j == len(projection) - 1:
                    end = i + j
                    lines_indices.append(int(start + ((end - start) / 2)))
                    return lines_indices
            end = i + j
            lines_indices.append(int(start + ((end - start) / 2)))
            i = i + j
        else:
            i += 1
    return lines_indices


def segments_indices(projection):
    lines_indices = []
    i = 0
    start = 0
    end = 0
    while i < len(projection):
        if projection[i] == 0:
            start = i
            j = 0

            while projection[j + i] == 0 and j + i < len(projection):
                j += 1
                if i + j == len(projection) - 1:
                    end = i + j
                    lines_indices.append(int(start + ((end - start) / 2)))
                    return lines_indices
            end = i + j
            lines_indices.append(int(start + ((end - start) / 2)))
            i = i + j
        else:
            i += 1
    return lines_indices


def words_segmentation(img, lines):

    # line=img[lines[0]:lines[1],:]
    # projection=projection=np.sum(line,axis=0)
    # print(lines[1]-lines[0])
    # subword=np.array([1,1,1])
    # projection=np.convolve(projection,subword,'same')
    # indices=segments_indices(projection)
    # return indices
    words_rects = []
    for i in range(len(lines) - 1):
        line = img[lines[i] : lines[i + 1], :]
        projection = np.sum(line, axis=0)
        projection = np.convolve(projection, np.array([1, 1, 1]), "same")
        indices = segments_indices(projection)

        words_rects.append(indices)

    return words_rects


def character_segmentation(
    wordSkeleton, baselineIndex, maxChangeIndex, topIndex, bottomIndex
):

    # getting separation region indices

    separationIndices = np.where(
        wordSkeleton[maxChangeIndex, :-1] != wordSkeleton[maxChangeIndex, 1:]
    )[0]
    separationIndices = separationIndices[1 : separationIndices.shape[0] - 1]
    separations = separationIndices.copy()
    separationIndices = separationIndices.reshape(-1, 2)
    # word = wordSkeleton.copy()
    # word[:, separations] = 0.3
    # view = ImageViewer(word)
    # view.show()
    # getting cut indices
    vp = np.sum(wordSkeleton, axis=0)
    mvf = stats.mode(vp[vp != 0]).mode[0]
    vpAbove = np.sum(wordSkeleton[:baselineIndex, :])
    vpBelow = np.sum(wordSkeleton[baselineIndex + 1 :, :])
    cutIndices = []
    regionsAndCuts = []
    # all possible cut indices
    for i in range(separationIndices.shape[0]):
        midRegion = separationIndices[i, 0] + int(
            (separationIndices[i, 1] - separationIndices[i, 0]) / 2
        )
        # while vp[midRegion] > mvf and midRegion > separationIndices[i, 0]:
        #     midRegion -= 1
        # if midRegion - 1 == separationIndices[i, 0]:
        #     midRegion = separationIndices[i, 0] + int(
        #         (separationIndices[i, 1] - separationIndices[i, 0]) / 2
        #     )
        #     while vp[midRegion] > mvf and midRegion < separationIndices[i, 1]:
        #         midRegion += 1
        #     if midRegion + 1 == separationIndices[i, 1]:
        #         midRegion = separationIndices[i, 0] + int(
        #             (separationIndices[i, 1] - separationIndices[i, 0]) / 2
        #         )
        while vp[midRegion] > mvf and midRegion < separationIndices[i, 1]:
            midRegion += 1
        if midRegion + 1 == separationIndices[i, 1]:
            midRegion = separationIndices[i, 0] + int(
                (separationIndices[i, 1] - separationIndices[i, 0]) / 2
            )
        # cutIndices.append(midRegion)
        regionsAndCuts.append(
            [separationIndices[i, 0], separationIndices[i, 1], midRegion]
        )

        # 3ashan lamma el7oroof beyet2esem menha 7etta soghayara keda lwahdaha
        # if np.sum(wordSkeleton[:, cutIndices[-1] : midRegion]) > 3:
        #     if midRegion - cutIndices[-1] > 2:
        #         cutIndices.append(midRegion)

    print("max Change", maxChangeIndex)
    # filtering cuts
    i = 0
    length = separationIndices.shape[0]
    lastflag = 1
    while i < length:
        if i == length:
            break
        region = wordSkeleton[:, regionsAndCuts[i][0] + 1 : regionsAndCuts[i][1]]

        # if i == 0:
        #     viewer = ImageViewer(region)
        #     viewer.show()
        hpRegion = np.sum(region, axis=1)
        # case ii in paper
        if hpRegion[baselineIndex] == 0 and i == 0 and lastflag == 1:
            print("no baseline", i)
            if np.sum(region[baselineIndex + 1 :, :]) > np.sum(
                region[0:baselineIndex, :]
            ):
                print("below more than above")
                regionsAndCuts.pop(i)
                i -= 1
                length -= 1
        # case iii in paper
        # epic fail
        # elif (
        #     i == 0
        #     and lastflag == 1
        #     and np.sum(wordSkeleton[:baselineIndex, :])
        #     > np.sum(wordSkeleton[baselineIndex + 1, :])
        # ):
        #     # getting top left index wared gedan yekon ghalat
        #     topleft = baselineIndex
        #     for m in range(regionsAndCuts[i][2]):
        #         for n in range(baselineIndex):
        #             if wordSkeleton[n, m] != 0:
        #                 topleft = n
        #                 break
        #     print("check case 3: topleft=", topleft)
        #     if baselineIndex - topleft < 0.5 * (baselineIndex - topIndex):
        #         print("case 3 successful")
        #         regionsAndCuts.pop(i)
        #         i -= 1
        #         length -= 1

        if i <= 0:
            lastflag = 0
        i += 1

    # strokes detection
    regionsAndCuts = np.asarray(regionsAndCuts)
    cutIndices = regionsAndCuts[:, 2]
    cutIndices = list(dict.fromkeys(cutIndices))
    # adding cut indices at beginning and end to be able to cut image
    cutIndices.insert(0, 0)
    if np.sum(wordSkeleton[:, cutIndices[-1] : wordSkeleton.shape[1]]) > 3:
        cutIndices.append(wordSkeleton.shape[1] - 1)
    # removing small characters

    length = len(cutIndices) - 1
    z = 0
    while z < length:
        if z == length:
            break
        if np.sum(wordSkeleton[:, cutIndices[z]+1 : cutIndices[z + 1]]) < 4:
            print("smaller then 4")
            cutIndices.pop(z)
            z -= 1
            length -= 1
        elif cutIndices[z + 1] - cutIndices[z] < 4:
            print("too tiny")
            cutIndices.pop(z)
            z -= 1
            length -= 1
        elif np.sum(wordSkeleton[:, cutIndices[z]+1 : cutIndices[z + 1]]) < 7 and len(np.sum(wordSkeleton[:, cutIndices[z]+1 : cutIndices[z + 1]],axis=0)[np.sum(wordSkeleton[:, cutIndices[z]+1 : cutIndices[z + 1]],axis=0)!=0]) ==1:
            print("short single line")
            cutIndices.pop(z)
            z -= 1
            length -= 1
        z += 1

    print("baselineindex", baselineIndex)
    strokesIndices = []
    length = len(cutIndices) - 1

    # for i in range(length):
    #     segment = wordSkeleton[:, cutIndices[i] : cutIndices[i + 1]]
    #     sumTopProjection = (np.sum(segment[0:baselineIndex, :], axis=1)).sum()
    #     sumBottomProjection = (np.sum(segment[baselineIndex + 1 :, :], axis=1)).sum()

    #     if sumTopProjection > sumBottomProjection:
    #         vp = np.sum(segment[:baselineIndex, :], axis=0)
    #         strokesHeight = np.max(vp)
    #         print(cutIndices[i], "passed condition1")
    #         h = np.sort(np.sum(segment, axis=1))[::-1][0]

    #         if strokesHeight <= h:
    #             print(cutIndices[i], "passed condition2 strokes Height=", strokesHeight)
    #             hp = np.sum(segment[:baselineIndex, :], axis=1)
    #             hp = hp[hp != 0]

    #             if stats.mode(hp).mode[0] == mvf:
    #                 print(cutIndices[i], "passed condition3")
    #                 strokesIndices.append(i)
    #         elif len(strokesIndices) >= 2:
    #             if i - strokesIndices[-1] == 1 and i - strokesIndices[-2] == 2:
    #                 if strokesHeight <= 5:
    #                     print(cutIndices[i], "passed condition4")
    #                     hp = np.sum(segment[:baselineIndex, :], axis=1)
    #                     hp = hp[hp != 0]

    #                     if stats.mode(hp).mode[0] == mvf:
    #                         print(cutIndices[i], "passed condition5")
    #                         strokesIndices.append(i)

    strokes = []
    # for i in range(len(strokesIndices)):
    #     strokes.append(cutIndices[strokesIndices[i]])
    # check that last letter is not split

    # make seen one letter instead of 3

    # if len(strokesIndices) > 2:
    #     i = len(strokesIndices) - 1
    #     while i >= 2:
    #         #
    #         print(i)
    #         if (
    #             strokesIndices[i] - strokesIndices[i - 1] == 1
    #             and strokesIndices[i - 1] - strokesIndices[i - 2] == 1
    #         ):
    #             print("popping")
    #             cutIndices.pop(strokesIndices[i])
    #             cutIndices.pop(strokesIndices[i - 1])

    #             i -= 2
    #         i -= 1

    return strokes, cutIndices
