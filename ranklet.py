import scipy.io
import time
import pdb
import numpy as np
from SSDL_GU import *
from sklearn.decomposition import SparseCoder
from numpy.linalg import norm
import sys
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from mnist import MNIST
from PIL import Image
import math
import os
import cv2

def Cal_R(rank_part):
    wid,hei=rank_part.shape
    m=n=wid*hei//2
    Rv=(rank_part[:,hei//2:].sum()-m*(m+1)//2)/(m*n//2)-1
    Rh=(rank_part[:wid//2,:].sum()-m*(m+1)//2)/(m*n//2)-1
    Rd=((rank_part[:wid//2,:hei//2].sum()+rank_part[wid//2:,hei//2:].sum())-m*(m+1)//2)/(m*n//2)-1
    return Rv,Rh,Rd

def Ranklet_Transform(img):
    wid,hei=img.shape
    rank_help=np.zeros(256,dtype=int)
    rank=np.empty((wid,hei),dtype=int)
    for i in range(wid):
        for j in range(hei):
            rank_help[img[i][j]]+=1
    for i in range(255):
        rank_help[i+1]+=rank_help[i]
    for i in range(wid):
        for j in range(hei):
            rank[wid-i-1][hei-j-1]=rank_help[img[wid-i-1][hei-j-1]]
            rank_help[img[wid-i-1][hei-j-1]]-=1
    Rvs=np.empty(75)
    Rhs=np.empty(75)
    Rds=np.empty(75)
    Rvs[0],Rhs[0],Rds[0]=Cal_R(rank)
    start=1
    scan_num=4
    for i in range(scan_num+1):
        for j in range(scan_num+1):
            Rvs[start],Rhs[start],Rds[start]=Cal_R(rank[i:wid-scan_num+i,j:hei-scan_num+j])
            start+=1
    scan_num=6
    for i in range(scan_num+1):
        for j in range(scan_num+1):
            Rvs[start],Rhs[start],Rds[start]=Cal_R(rank[i:wid-scan_num+i,j:hei-scan_num+j])
            start+=1
    return Rvs,Rhs,Rds

def Cal_Feature(ranklet):
    N=ranklet.shape[0]
    Ei=ranklet.sum()/N
    minus_Ei=ranklet-Ei
    sigma=math.sqrt((minus_Ei*minus_Ei).sum()/N)
    s=np.cbrt((minus_Ei*minus_Ei*minus_Ei).sum()/N)
    return Ei,sigma,s

def Extract_Feature(img):
    Rvs,Rhs,Rds=Ranklet_Transform(img)
    feature_part=np.empty(9)
    feature_part[0],feature_part[1],feature_part[2]=Cal_Feature(Rvs)
    feature_part[3],feature_part[4],feature_part[5]=Cal_Feature(Rhs)
    feature_part[6],feature_part[7],feature_part[8]=Cal_Feature(Rds)
    return feature_part

im=Image.open('./0a371857f68631e2b63043179a11777d.jpg')
img=np.array(im)
img=img.transpose((1,0,2))
feature=np.empty(27)
for i in range(3):
    feature[i*9:(i+1)*9]=Extract_Feature(img[:,:,i])
pdb.set_trace()