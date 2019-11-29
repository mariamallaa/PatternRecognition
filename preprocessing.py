import numpy as np
from scipy.ndimage import interpolation as inter
import cv2
def find_score(arr, angle):
    data = inter.rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return score

def adjust_tilt(img):
    bin_img=img.copy()
    bin_img = 1 - (bin_img / 255.0)

    delta = 1
    limit = 5
    angles = np.arange(-limit, limit+delta, delta)
    scores = []
    for angle in angles:
        score = find_score(bin_img, angle)
        scores.append(score)

    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]
    corrected_img = inter.rotate(bin_img, best_angle, reshape=False, order=0)

    return corrected_img



def line_segmentation(img):
    projection=np.sum(img,axis=1)
    print(projection[21])
    lines_indices=[]
    i=0
    start=0
    end=0
    while i < len(projection):
        if projection[i]==0:
            start=i
            j=0

            while projection[j+i]==0 and j+i<len(projection):
                j+=1
                if i+j==len(projection)-1:
                    end=i+j
                    lines_indices.append(int(start+((end-start)/2)))
                    return lines_indices
            end=i+j
            lines_indices.append(int(start+((end-start)/2)))
            i=i+j
        else:
            i+=1
    return lines_indices

# def words_separation(img):
