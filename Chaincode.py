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
import statistics
from skimage.morphology import square
from skimage import feature

# from preprocessing import *
from test import *
import cv2
from skimage.morphology import thin, skeletonize,binary_dilation,binary_erosion,binary_closing,binary_opening
from skimage.transform import resize
from scipy import stats
from skimage.measure import label

def chain_code(image):
    output=[]
    output = cv2.connectedComponentsWithStats(image.astype('uint8'), 4)
    stats=output[2]
    m=np.array(output[0]-1)
    m=stats[1::,4]
    ind = np.argmax(m)
    start=stats[ind+1][1]
    chain=[]
    currenti=0
    currentj=0
    dir=[[0,1],[-1,0],[0,-1],[1,0]]
    starti=0
    startj=0
    i=0
    j=0
    p=0
    for j in range(image.shape[1]):
            if(image[start][j]):
                currenti=start
                currentj=j
                starti=start
                startj=j
                break        
    i=3
    indexi=0
    indexj=0
    g=0
    while(1):
        if(i==4):
            i=0
        if(image[int(currenti+dir[i][0])][int(currentj+dir[i][1])]):
            chain.append(i)
            currenti=currenti+dir[i][0]
            currentj=currentj+dir[i][1]
            if(currenti==starti and currentj==startj):
                break
            i=(i+3)%4
            continue

        elif(image[(currenti+dir[i][0])][(currentj+dir[i][1])]==0):
            i+=1
            continue

    return(chain)


def chain_code7(image):
    output=[]
    output = cv2.connectedComponentsWithStats(image.astype('uint8'), 4)
    stats=output[2]
    m=np.array(output[0]-1)
    m=stats[1::,4]
    ind = np.argmax(m)
    start=stats[ind+1][1]
    chain=[]
    currenti=0
    currentj=0
    dir=[[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1],[1,0],[1,1]]
    starti=0
    startj=0
    i=0
    j=0
    p=0
    for j in range(image.shape[1]):
            if(image[start][j]):
                currenti=start
                currentj=j
                starti=start
                startj=j
                break        
    i=6
    indexi=0
    indexj=0
    while(1):
        if(i==8):
            i=0
        
        if(image[int(currenti+dir[i][0])][int(currentj+dir[i][1])]):
            chain.append(i)
            print(int(currenti+dir[i][0]),int(currentj+dir[i][1]))
            currenti=currenti+dir[i][0]
            currentj=currentj+dir[i][1]
            if(currenti==starti and currentj==startj):
                break
            i=(i+7)%8
            continue

        elif(image[(currenti+dir[i][0])][(currentj+dir[i][1])]==0):
            i+=1
            continue

    return(chain)

def chain_code4(h):
    chain=[]
    currenti=0
    currentj=0
    dir=[[0,1],[-1,0],[0,-1],[1,0]]
    starti=0
    startj=0
    i=0
    j=0
    p=0
    print(h)
    for i in range(h.shape[0]):
        if(p==27):
            break
        for j in range(h.shape[1]):
            if(h[i][j]):
                currenti=i
                currentj=j
                starti=i
                startj=j
                p=27
                break
        
                
    print(currenti,currentj)        
    i=3
    indexi=0
    indexj=0
    #print(currenti,currentj)
    g=0
    while(1):
        if(i==4):
            i=0
        '''
        if(int(currenti+dir[i][0])==30 or int(currentj+dir[i][1])==30):
            break
        '''
        #print(chain)
        
        if(h[int(currenti+dir[i][0])][int(currentj+dir[i][1])]):
            chain.append(i)
            #print(int(currenti+dir[i][0]),int(currentj+dir[i][1]))
            currenti=currenti+dir[i][0]
            currentj=currentj+dir[i][1]
            if(currenti==starti and currentj==startj):
                break
            i=(i+3)%4
            continue

        elif(h[(currenti+dir[i][0])][(currentj+dir[i][1])]==0):
            i+=1
            continue

    return(chain)


def Get_connected_comp(resized):
    output = cv2.connectedComponentsWithStats(resized.astype('uint8'), 4)
    number_of_holes=output[0]-1
    return number_of_holes