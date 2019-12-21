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



def Get_connected_comp(resized):
    output = cv2.connectedComponentsWithStats(resized.astype('uint8'), 4)
    number_of_holes=output[0]-1
    return number_of_holes


def Find_holes(binary):
    holes=[]
    _,contours, hierarchy=cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

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

<<<<<<< Updated upstream
def chain_code_t(h,k):
=======
def chain_code_t(h):
>>>>>>> Stashed changes
    
    output=[]

    img=np.zeros([h.shape[0]+1,h.shape[1]+1])

    img[1:img.shape[0],1:img.shape[1]]=h

    img[0,:]=0

    img[:,0]=0

    img[h.shape[0],:]=0

    img[:,h.shape[1]]=0

    output = cv2.connectedComponentsWithStats(img.astype('uint8'), 8)

    stats=output[2]

    m=np.array(output[0]-1)

<<<<<<< Updated upstream
    print(stats)
=======
    #print(stats)
>>>>>>> Stashed changes

    m=stats[1::,4]
    
    labels=output[1]
    #print(labels[0:27,0:27].astype('uint8'))
 

    if(len(m)==0):

        chain=[-1]*70

        return chain

    ind = np.argmax(m)

    x=stats[ind+1][1]

    y=stats[ind+1][0]

    for i in range(1,len(stats),1):

        if stats[i][1]==x and stats[i][0]==y and i!=ind+1:

            chain=[-1]*70

            return chain

 

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
    

    for j in range(img.shape[1]):

            if(labels[start][j]==ind+1):
                

                currenti=start

                currentj=j

                starti=start

                startj=j

                break           

    i=3

    chain.append(i)

    indexi=0

    indexj=0

    while(1):

        if(i==4):

            i=0

        if(currentj==0):

            i=3

        if(img[int(currenti+dir[i][0])][int(currentj+dir[i][1])]):

            chain.append(i)

            currenti=currenti+dir[i][0]

            currentj=currentj+dir[i][1]

            i=(i+3)%4

            g+=1

            if(g==69):

                break

            if(currenti==starti and currentj==startj):

                break

            if(currentj==0):

                i=3

            if(currentj==img.shape[1]-1):

                i=1

            continue

 

        elif(img[(currenti+dir[i][0])][(currentj+dir[i][1])]==0):

            i+=1

            continue

    if(g<69):

        for i in range(69-g):

            chain.append(-1)

    return(chain)


def Get_secondary(binary,secondary):
    output=[]
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


def Get_main(binary,main):
    output=[]
    output = cv2.connectedComponentsWithStats(binary.astype('uint8'), 4)
    stats=output[2]
    m=np.array(output[0]-1)
    m=stats[1::,4]
    indmax = np.argmax(m)
    #area
    main.append(stats[indmax+1][4])
    #w/h
    main.append(stats[indmax+1][2]/stats[indmax+1][3])
    return(main)

def Distribution(binary,main):
    output=[]
    output = cv2.connectedComponentsWithStats(binary.astype('uint8'), 4)
    stats=output[2]
    m=np.array(output[0]-1)
    m=stats[1::,4]
    indmax = np.argmax(m)
    A=stats[indmax+1][4]
    main.append(np.round(np.sum(binary[0:14,0:14])/A,2))
    main.append(np.round(np.sum(binary[0:14,14:28]/A),2))
    main.append(np.round(np.sum(binary[14:28,0:14]/A),2))
    main.append(np.round(np.sum(binary[14:28,14:28]/A),2))
    return(main)


<<<<<<< Updated upstream
def Combine(binary,k):
    
    print(k)
    features=chain_code_t(binary,k)
    print("finished chaincode")
    featuresgetsec=Get_secondary(binary,features)
    print("finish getsec")
    featuresgetsec.append(Find_holes(binary))
    print("find holes")
    featuresgetsec=Get_main(binary,featuresgetsec)
    print("find mains")
    featuresgetsec=Distribution(binary,featuresgetsec)
    print("dist")
=======
def Combine(binary):
    
    #print(k)
    features=chain_code_t(binary)
    #print("finished chaincode")
    featuresgetsec=Get_secondary(binary,features)
    #print("finish getsec")
    featuresgetsec.append(Find_holes(binary))
    #print("find holes")
    featuresgetsec=Get_main(binary,featuresgetsec)
    #print("find mains")
    featuresgetsec=Distribution(binary,featuresgetsec)
    #print("dist")
>>>>>>> Stashed changes
    
    return(featuresgetsec)