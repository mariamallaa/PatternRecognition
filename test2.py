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


def Find_holes(binary):
    holes = []
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for i in range(len(hierarchy[0])):
        if hierarchy[0][i][3] >= 0:
            holes = contours[i]
    if len(holes) == 0:
        num_holes = 0
    else:
        mask = np.zeros(binary.shape, np.uint8)
        cv2.drawContours(mask, holes, -1, 1, cv2.FILLED)
        num_holes = Get_connected_comp(mask)
    return num_holes


def Get_connected_comp(resized):
    output = cv2.connectedComponentsWithStats(resized.astype("uint8"), 4)
    number_of_holes = output[0] - 1
    return number_of_holes


def character_segmentation(
    word, wordSkeleton, baselineIndex, maxChangeIndex, topIndex, bottomIndex
):
    # wordSkeleton = word.copy()
    print("line baseline", baselineIndex)
    # word baseline
    projection = np.sum(wordSkeleton, axis=1)
    baselineIndex = max(baselineIndex, np.argmax(projection))
    # max vertical change word
    # verticalChange = []
    # for k in range(baselineIndex):
    #     verticalChange.append(
    #         len(np.where(wordSkeleton[k, :-1] != wordSkeleton[k, 1:])[0])
    #     )
    # verticalChange = np.asarray(verticalChange)
    # maxChangeIndex = max(np.argmax(verticalChange), baselineIndex - 3)

    print("word baseline:", baselineIndex, "maxChange word", maxChangeIndex)
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

    # checks on last letter
    if separationIndices.shape[0] > 1 and separationIndices.shape[1] > 1:
        # case ii in paper
        region = wordSkeleton[:, separationIndices[0, 1] : separationIndices[1, 0]]
        if np.sum(region[baselineIndex + 1 :, :]) > np.sum(region[0:baselineIndex, :]):
            print("below more than above")
            region = region[:, 1:]
            hpRegion = np.sum(region, axis=1)
            if hpRegion[baselineIndex] == 0:
                if vp[cutIndices[1]] != 0:
                    print("case 2 successful")
                    cutIndices.pop(1)
        # case iii in paper

        elif np.sum(region[:baselineIndex, :]) > np.sum(region[baselineIndex + 1, :]):
            # getting top left index wared gedan yekon ghalat
            # region = wordSkeleton[:, regionsAndCuts[i][0] - 3 : regionsAndCuts[i][1]]
            stop = baselineIndex
            hpRegion = np.sum(region, axis=1)
            if hpRegion[baselineIndex] < hpRegion[baselineIndex - 1]:
                stop = baselineIndex - 1

            start = 0
            vpRegion = np.sum(region, axis=0)
            if vpRegion[0] > hpRegion[1]:
                start = 1

            breakflag = 0
            topleft = 0
            for m in range(start, region.shape[1], 1):
                for n in range(stop):
                    print("column", separationIndices[0, 1] + m, "row", n)
                    if region[n, m] != 0:
                        topleft = n
                        breakflag = 1
                        break
                if breakflag == 1:
                    break
            print("start", start + separationIndices[0, 1], "stop", stop)
            print(
                "check case 3: topleft=",
                topleft,
                "start index=",
                separationIndices[0, 1],
            )
            if baselineIndex - topleft < 0.5 * (baselineIndex - topIndex):
                if vp[cutIndices[1]] != 0:
                    print("case 3 successful")
                    cutIndices.pop(1)

    # strokes detection

    print(baselineIndex)
    strokesIndices = []
    length = len(cutIndices) - 1

    for i in range(length):
        segment = wordSkeleton[:, cutIndices[i] : cutIndices[i + 1]]
        sumTopProjection = (np.sum(segment[0:baselineIndex, :], axis=1)).sum()
        sumBottomProjection = (np.sum(segment[baselineIndex + 1 :, :], axis=1)).sum()

        if sumTopProjection > sumBottomProjection and vp[cutIndices[i]] != 0:
            vpSegment = np.sum(segment[:baselineIndex, :], axis=0)
            strokesHeight = np.max(vpSegment)
            print(cutIndices[i], "passed condition1")
            h = np.sort(np.sum(segment, axis=1))[::-1][0]

            if strokesHeight <= h:
                print(cutIndices[i], "passed condition2 strokes Height=", strokesHeight)
                hp = np.sum(segment[:baselineIndex, :], axis=1)
                hp = hp[hp != 0]

                if stats.mode(hp).mode[0] == mvf:
                    print(cutIndices[i], "passed condition3")
                    strokesIndices.append(i)
            elif len(strokesIndices) >= 2:
                if i - strokesIndices[-1] == 1 and i - strokesIndices[-2] == 2:
                    if strokesHeight <= 5:
                        print(cutIndices[i], "passed condition4")
                        hp = np.sum(segment[:baselineIndex, :], axis=1)
                        hp = hp[hp != 0]

                        if stats.mode(hp).mode[0] == mvf:
                            print(cutIndices[i], "passed condition5")
                            strokesIndices.append(i)

    strokes = []
    for i in range(len(strokesIndices)):
        strokes.append(cutIndices[strokesIndices[i]])

    if len(strokesIndices) > 2:
        i = len(strokesIndices) - 1
        while i >= 2:
            #
            print(i)
            if (
                strokesIndices[i] - strokesIndices[i - 1] == 1
                and strokesIndices[i - 1] - strokesIndices[i - 2] == 1
            ):
                if (
                    cv2.connectedComponentsWithStats(
                        word[
                            :,
                            cutIndices[strokesIndices[i]] : cutIndices[
                                strokesIndices[i] + 1
                            ],
                        ].astype("uint8"),
                        8,
                    )[0]
                    == 2
                    and cv2.connectedComponentsWithStats(
                        word[
                            :,
                            cutIndices[strokesIndices[i - 1]] : cutIndices[
                                strokesIndices[i]
                            ],
                        ].astype("uint8"),
                        8,
                    )[0]
                    == 2
                    and cv2.connectedComponentsWithStats(
                        word[
                            :,
                            cutIndices[strokesIndices[i - 2]] : cutIndices[
                                strokesIndices[i - 1]
                            ],
                        ].astype("uint8"),
                        8,
                    )[0]
                    == 2
                ):

                    # print("connected components:", output[0])
                    print("popping")

                    cutIndices.pop(strokesIndices[i])
                    cutIndices.pop(strokesIndices[i - 1])
                    strokesIndices.pop(i)
                    strokesIndices.pop(i - 1)
                    strokesIndices.pop(i - 2)

                    i -= 2
                elif (
                    cv2.connectedComponentsWithStats(
                        word[
                            :,
                            cutIndices[strokesIndices[i]] : cutIndices[
                                strokesIndices[i] + 1
                            ],
                        ].astype("uint8"),
                        8,
                    )[0]
                    + cv2.connectedComponentsWithStats(
                        word[
                            :,
                            cutIndices[strokesIndices[i - 1]] : cutIndices[
                                strokesIndices[i]
                            ],
                        ].astype("uint8"),
                        8,
                    )[0]
                    + cv2.connectedComponentsWithStats(
                        word[
                            :,
                            cutIndices[strokesIndices[i - 2]] : cutIndices[
                                strokesIndices[i - 1]
                            ],
                        ].astype("uint8"),
                        8,
                    )[0]
                    == 8
                ):

                    print("sheen successful")
                    print("popping")

                    cutIndices.pop(strokesIndices[i])
                    cutIndices.pop(strokesIndices[i - 1])

                    strokesIndices.pop(i)
                    strokesIndices.pop(i - 1)
                    strokesIndices.pop(i - 2)

                    i -= 2

            i -= 1

    # i = len(strokesIndices) - 1
    # while i >= 0:
    #     if (
    #         len(strokesIndices) > 0
    #         and len(cutIndices) > 2
    #         and len(cutIndices) > len(strokesIndices)
    #         and len(cutIndices) > strokesIndices[i] + 1
    #     ):
    #         if (
    #             Find_holes(
    #                 word[
    #                     :,
    #                     cutIndices[strokesIndices[i]] : cutIndices[
    #                         strokesIndices[i] + 1
    #                     ],
    #                 ].astype("uint8")
    #             )
    #             == 1
    #         ):
    #             cutIndices.pop(strokesIndices[i])
    #             print("popped dad")

    #         i -= 1
    #     else:
    #         break

    i = len(strokesIndices) - 1
    while i >= 0:
        if (
            len(strokesIndices) > 0
            and len(cutIndices) > 2
            and len(cutIndices) > len(strokesIndices)
            and len(cutIndices) > strokesIndices[i] + 2
        ):

            if (
                cv2.connectedComponentsWithStats(
                    word[
                        :,
                        cutIndices[strokesIndices[i]] : cutIndices[
                            strokesIndices[i] + 1
                        ],
                    ].astype("uint8"),
                    8,
                )[0]
                == 2
                and Find_holes(
                    word[
                        :,
                        cutIndices[strokesIndices[i] + 1] : cutIndices[
                            strokesIndices[i] + 2
                        ],
                    ].astype("uint8")
                )
                == 1
            ):
                if vp[cutIndices[strokesIndices[i] + 1]] != 0:
                    cutIndices.pop(strokesIndices[i] + 1)
                    print("sad or dad popped")
            i -= 1

        else:
            break

    return strokes, cutIndices
