import numpy as np
from scipy.ndimage import interpolation as inter
import cv2
def find_score(arr, angle):
    data = inter.rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return score

#using for loops we will not use it IGNORE
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

def correct_skew(img):
    thresh=img.copy()
    thresh=1-(thresh/255)

    coords = np.column_stack(np.where(thresh > 0))

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle=-angle

    
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    rotated=1-(rotated/255)
    return rotated

def line_segmentation(img):
    projection=np.sum(img,axis=1)
    
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

def segments_indices(projection):
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

def words_segmentation(img, lines):
    
    # line=img[lines[0]:lines[1],:]
    # projection=projection=np.sum(line,axis=0)
    # print(lines[1]-lines[0])
    # subword=np.array([1,1,1])
    # projection=np.convolve(projection,subword,'same')
    # indices=segments_indices(projection)
    # return indices
    words_rects=[]
    for i in range(len(lines)-1):
        line=img[lines[i]:lines[i+1],:]
        projection=np.sum(line,axis=0)
        projection=np.convolve(projection,np.array([1,1]),'same')
        indices=segments_indices(projection)
       
        words_rects.append(indices)
    
    
    return words_rects

        

