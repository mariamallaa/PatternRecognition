import numpy as np
from scipy.ndimage import interpolation as inter
from scipy import stats
import cv2
from skimage.viewer import ImageViewer


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
    word, wordSkeleton, baselineIndex, maxChangeIndex, topIndex, bottomIndex
):

    # projection = np.sum(wordSkeleton, axis=1)
    # baselineIndex = max(baselineIndex, np.argmax(projection))
    # getting separation region indices

    separationIndices = np.where(
        wordSkeleton[maxChangeIndex, :-1] != wordSkeleton[maxChangeIndex, 1:]
    )[0]
    separationIndices = separationIndices[1 : separationIndices.shape[0] - 1]
    separationIndices = separationIndices.reshape(-1, 2)

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
        vpRegion = np.sum(
            wordSkeleton[:, separationIndices[i, 0] : separationIndices[i, 1]], axis=0
        )
        zeros = np.where(vpRegion == 0)[0]
        if len(zeros) > 0:

            nearest = np.absolute(midRegion - zeros)

            midRegion = separationIndices[i, 0] + zeros[np.argmin(nearest)]

        else:
            while vp[midRegion] > mvf and midRegion > separationIndices[i, 0]:
                midRegion -= 1
            if midRegion == separationIndices[i, 0]:
                midRegion = separationIndices[i, 0] + int(
                    (separationIndices[i, 1] - separationIndices[i, 0]) / 2
                )
                while vp[midRegion] > mvf and midRegion < separationIndices[i, 1]:
                    midRegion += 1

            # if midRegion + 1 == separationIndices[i, 1]:
            #     midRegion = separationIndices[i, 0] + int(
            #         (separationIndices[i, 1] - separationIndices[i, 0]) / 2
            #     )
        if midRegion != separationIndices[i, 1]:
            regionsAndCuts.append(
                [separationIndices[i, 0], separationIndices[i, 1], midRegion]
            )
        # regionsAndCuts.append(
        #     [separationIndices[i, 0], separationIndices[i, 1], midRegion]
        # )

    if len(regionsAndCuts) == 0:
        return [], [0, wordSkeleton.shape[1] - 1]

    regionsAndCuts = np.asarray(regionsAndCuts)

    # filtering cuts
    # hole detection shaghala bs mesh 3ayzenha
    # i = 1
    # length = len(regionsAndCuts) - 1
    # while i < length:
    #     if i >= length:
    #         break
    #     char = wordSkeleton[:, regionsAndCuts[i - 1, 2] : regionsAndCuts[i, 2]]
    #     if Find_holes(char.astype("uint8")) != 0:
    #         print("hole found")
    #         regionsAndCuts = np.delete(regionsAndCuts, i - 1, 0)
    #         length -= 1

    #     i += 1

    if len(regionsAndCuts) > 0:
        region = wordSkeleton[:, regionsAndCuts[0, 0] : regionsAndCuts[0, 1]]
        if np.sum(region[baselineIndex + 1 :, :]) > np.sum(region[0:baselineIndex, :]):

            region = region[:, 1:]
            hpRegion = np.sum(region, axis=1)
            if hpRegion[baselineIndex] == 0:
                # print("no baseline")
                if vp[regionsAndCuts[0, 2]] != 0:
                    print("case 2 successful")
                    regionsAndCuts = np.delete(regionsAndCuts, 0, 0)
        elif np.sum(region[baselineIndex + 1 :, :]) < np.sum(
            region[0:baselineIndex, :]
        ):

            stop = baselineIndex
            hpRegion = np.sum(region, axis=1)
            # if hpRegion[baselineIndex] < hpRegion[baselineIndex - 1]:
            #     stop = baselineIndex - 1

            start = 0
            breakflag = 0
            topleft = 0
            for m in range(start, region.shape[1], 1):
                for n in range(stop):
                    # print("column", regionsAndCuts[0, 0] + m, "row", n)
                    if region[n, m] != 0:
                        topleft = n
                        breakflag = 1
                        break
                if breakflag == 1:
                    break
            # print("start", start + regionsAndCuts[0, 0], "stop", stop)
            # print(
            #     "check case 3: baseline=",
            #     baselineIndex,
            #     "stop=",
            #     stop,
            #     "topleft=",
            #     m,
            #     ",",
            #     topleft,
            #     "start index=",
            #     regionsAndCuts[0, 0],
            # )
            if baselineIndex - topleft < 0.5 * (baselineIndex - topIndex):
                print("Smaller->vp at cut=", vp[regionsAndCuts[0, 2]], "and mfv=", mvf)
                # if vp[regionsAndCuts[0, 2]] != 0 and vp[regionsAndCuts[0, 2]] != mvf:
                if (
                    vp[regionsAndCuts[0, 2]] != 0
                    and np.sum(
                        wordSkeleton[baselineIndex + 1 :, 0 : regionsAndCuts[0, 2]]
                    )
                    == 0
                ):
                    print("case 3 successful")
                    regionsAndCuts = np.delete(regionsAndCuts, 0, 0)

    cutIndices = regionsAndCuts[:, 2]
    cutIndices = np.insert(cutIndices, 0, 0)
    cutIndices = np.append(cutIndices, wordSkeleton.shape[1] - 1)

    # filtering small cut segments

    i = 0
    length = len(cutIndices) - 1
    while i < length:
        if i >= length:
            break
        if (
            cutIndices[i + 1] - cutIndices[i] <= 3
            or np.sum(wordSkeleton[:, cutIndices[i] : cutIndices[i + 1]]) <= 3
        ):
            print("Removed unnecessary cuts")
            cutIndices = np.delete(cutIndices, i + 1)
            length -= 1
        i += 1

    # getting strokes
    strokesIndices = []
    for i in range(len(cutIndices) - 1, 0, -1):
        segment = wordSkeleton[:, cutIndices[i - 1] : cutIndices[i]]
        if cv2.connectedComponentsWithStats(segment.astype("uint8"), 8,)[0] == 2:
            print("single connected compnent")
            if np.sum(segment[:baselineIndex, :]) > np.sum(
                segment[baselineIndex + 1 :, :]
            ):
                print("below more than above")
                hpSegment = np.sum(segment, axis=1)
                vpSegment = np.sum(segment, axis=0)
                width = len(vpSegment[vpSegment != 0])
                if np.max(np.sum(segment, axis=0)) <= 1.5 * width:
                    print("height within range")
                    hpSegment = hpSegment[hpSegment != 0]

                    if stats.mode(hpSegment).mode[0] == mvf:
                        print("thickness equal mode value")
                        if Find_holes(segment.astype("uint8")) == 0:
                            print("no holes")
                            if vpSegment[0] != 0:
                                print("not dal, successful stroke!")
                                strokesIndices.append(i)
    print(strokesIndices)

    strokes = []
    for i in range(len(strokesIndices)):
        strokes.append(cutIndices[strokesIndices[i]])

    if len(strokesIndices) > 2:

        length = len(strokesIndices)
        i = 0
        while i <= length - 3:
            if i > length - 3:
                break
            if (
                strokesIndices[i] - strokesIndices[i + 1] == 1
                and strokesIndices[i] - strokesIndices[i + 2] == 2
            ):
                print("popping seen")
                cutIndices = np.delete(
                    cutIndices, [strokesIndices[i + 1], strokesIndices[i + 2]]
                )
                i += 2
            i += 1
    if len(strokesIndices) > 1:
        length = len(strokesIndices)
        i = 0
        while i <= length - 2:
            if i > length - 2:
                break

            if strokesIndices[i] - strokesIndices[i + 1] == 2:
                print("possible sheen")

                if (
                    cv2.connectedComponentsWithStats(
                        wordSkeleton[
                            :,
                            cutIndices[strokesIndices[i + 1]] : cutIndices[
                                strokesIndices[i + 1] + 1
                            ],
                        ].astype("uint8"),
                        8,
                    )[0]
                    >= 3
                ):
                    print("popping sheen")
                    print(
                        "removing Indices", strokesIndices[i] - 1, strokesIndices[i + 1]
                    )
                    cutIndices = np.delete(
                        cutIndices, [strokesIndices[i] - 1, strokesIndices[i + 1]]
                    )
                    i += 1
            i += 1
    if len(strokesIndices) > 0:
        i = 0
        while i < len(strokesIndices):
            if ( len(strokesIndices) > 0 and len(cutIndices) > 2 and len(cutIndices) > len(strokesIndices) and len(cutIndices) > strokesIndices[i] + 2):
                if cutIndices[strokesIndices[i]] != wordSkeleton.shape[1] - 1:
                    previousSegment = wordSkeleton[
                        :, cutIndices[strokesIndices[i]] : cutIndices[strokesIndices[i] + 1]
                    ]
                    if Find_holes(previousSegment.astype("uint8")) != 0:
                        print("popped sad")
                        cutIndices = np.delete(cutIndices, strokesIndices[i])
                i += 1
            else:
                break
    # return [], regionsAndCuts[:, 2]
    return strokes, cutIndices

