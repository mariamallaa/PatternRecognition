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



def Get_main(binary):
    output=[]
    main=[]
    output = cv2.connectedComponentsWithStats(binary.astype('uint8'), 4)
    stats=output[2]
    m=np.array(output[0]-1)
    m=stats[1::,4]
    print(stats)
    indmax = np.argmax(m)
    #area
    main.append(stats[indmax+1][4])
    #w/h
    main.append(stats[indmax+1][2]/stats[indmax+1][3])
    return(main)

def Get_secondary(binary):
    output=[]
    secondary=[]
    output = cv2.connectedComponentsWithStats(binary.astype('uint8'), 4)
    secondary.append(output[0]-2)
    stats=output[2]
    m=np.array(output[0]-1)
    m=stats[1::,4]
    indmax = np.argmax(m)
    indmin=np.argmin(m)
    start_main=stats[indmax+1][1]
    start_sec=stats[indmin+1][1]
    #below
    if(start_main<start_sec):
        secondary.append(-1)
        secondary.append(stats[indmin+1][4])
    #above
    elif(start_main>start_sec):
        secondary.append(1)
        secondary.append(stats[indmin+1][4])
    #no dots
    else:
        secondary.append(0)
        secondary.append(0)
    return(secondary)

def Projection(binary):

    V_proj= np.sum(binary, axis=0)
    H_proj= np.sum(binary, axis=1)
    return(V_proj,H_proj)

def Get_connected_comp(resized):
    output = cv2.connectedComponentsWithStats(resized.astype('uint8'), 4)
    number_of_holes=output[0]-1
    return number_of_holes


def Find_holes(binary):
    holes=[]
    contours, hierarchy=cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for i in range(len(hierarchy[0])):
        if hierarchy[0][i][3] >= 0:
            holes=contours[i]
    if(len(holes)==0):
        num_holes=0
    else:
        mask = np.zeros(binary.shape, np.uint8)
        cv2.drawContours(mask, holes, -1, 1, cv2.FILLED) 
        num_holes=Get_connected_comp(mask)
    return num_holes

def chain_code_test(h):
    output=[]
    output = cv2.connectedComponentsWithStats(h.astype('uint8'), 4)
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
    for j in range(h.shape[1]):
            if(h[start][j]):
                currenti=start
                currentj=j
                starti=start
                startj=j
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
        print(chain)
        print(int(currenti+dir[i][0]),int(currentj+dir[i][1]))
        if(h[int(currenti+dir[i][0])][int(currentj+dir[i][1])]):
            chain.append(i)
            currenti=currenti+dir[i][0]
            currentj=currentj+dir[i][1]
            i=(i+3)%4
            if(currenti==starti and currentj==startj):
                break
            if(currentj==0):
                i=3
            if(currentj==27):
                i=1
            continue

        elif(h[(currenti+dir[i][0])][(currentj+dir[i][1])]==0):
            i+=1
            continue

    return(chain)


def chain_code_t(h):
    output=[]
    output = cv2.connectedComponentsWithStats(h.astype('uint8'), 4)
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
    g=0
    for j in range(h.shape[1]):
            if(h[start][j]):
                currenti=start
                currentj=j
                starti=start
                startj=j
                break   
    print(currenti,currentj)        
    i=3
    chain.append(i)
    indexi=0
    indexj=0
    #print(currenti,currentj)
    
    while(1):
        if(i==4):
            i=0
        print(chain)
        print(int(currenti+dir[i][0]),int(currentj+dir[i][1]))
        if(h[int(currenti+dir[i][0])][int(currentj+dir[i][1])]):
            chain.append(i)
            currenti=currenti+dir[i][0]
            currentj=currentj+dir[i][1]
            i=(i+3)%4
            g+=1
            if(g==80):
                break
            if(currenti==starti and currentj==startj):
                break
            if(currentj==0):
                i=3
            if(currentj==h.shape[1]-1):
                i=1
            continue

        elif(h[(currenti+dir[i][0])][(currentj+dir[i][1])]==0):
            i+=1
            continue
    if(g<80):
        for i in range(80-g):
            chain.append(-1)
    return(chain)


def Distribution(binary):
    output=[]
    main=[]
    output = cv2.connectedComponentsWithStats(binary.astype('uint8'), 4)
    stats=output[2]
    m=np.array(output[0]-1)
    m=stats[1::,4]
    indmax = np.argmax(m)
    A=stats[indmax+1][4]
    h=binary.shape[0]
    w=binary.shape[1]
    h2=math.floor(h/2)
    w2=math.ceil(w/2)
    main.append(np.round(np.sum(binary[0:h2,0:w2])/A,2))
    main.append(np.round(np.sum(binary[0:h2,w2:w]/A),2))
    main.append(np.round(np.sum(binary[h2:h,0:w2]/A),2))
    main.append(np.round(np.sum(binary[h2:h,w2:w]/A),2))
    return(main)

def Combine(binary):
    features=[]
    features.append(chain_code_t(binary))
    features.append(Get_secondary(binary))
    features.append(Find_holes(binary))
    features.append(Get_main(binary))
    features.append(Distribution(binary))
    return(features)
