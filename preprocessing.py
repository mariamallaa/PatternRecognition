import numpy as np
from scipy.ndimage import interpolation as inter
from scipy import stats
import cv2


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


def character_segmentation(wordSkeleton):
    baselineIndex = np.argmax(np.sum(wordSkeleton, axis=1))
    # finding maximum transition index
    verticalChange = []
    for i in range(baselineIndex):
        verticalChange.append(
            len(np.where(wordSkeleton[i, :-1] != wordSkeleton[i, 1:])[0])
        )
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
    if len(strokesIndices) > 2:
        for i in range(len(strokesIndices) - 2):
            print(i)
            if (
                strokesIndices[i + 2] - strokesIndices[i + 1] == 1
                and strokesIndices[i + 1] - strokesIndices[i] == 1
            ):
                cutIndices.pop(strokesIndices[i + 2])
                cutIndices.pop(strokesIndices[i + 1])

                i += 2

    return cutIndices
