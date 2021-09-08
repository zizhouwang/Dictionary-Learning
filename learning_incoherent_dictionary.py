import scipy.io
import time
import pdb
import numpy as np
from SSDL_GU import *
from sklearn.decomposition import SparseCoder
from numpy.linalg import norm
from numpy import linalg as LA
import sys
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from mnist import MNIST
from PIL import Image
import math
import os
import cv2
import copy

def incoherent(the_D,the_Y,the_X,nIter):
    iIter=0
    u0=0.7
    current_u=1.
    while iIter<=nIter and current_u>u0:
        gram=np.dot(the_D,the_D.T)
        the_K=copy.deepcopy(gram)
        the_K_max=the_K.max()
        # u0=(1-the_K_max)/2.+the_K_max
        the_K[the_K<-u0]=-u0
        the_K[the_K>u0]=u0
        for i in range(the_K.shape[0]):
            the_K[i][i]=1.
        Q,eigenvalues=LA.eig(gram)
        eigenvalues[eigenvalues<0]=0
        the_D=np.dot(np.sqrt(eigenvalues),Q.T)
        the_C=np.dot(the_Y,np.dot(the_D,the_X).T)
        the_W=the_u,the_s,the_vt=LA.svd(the_C)
        np.dot(the_vt.T,the_u.T)
        the_D=np.dot(the_W,the_D)
        current_u=get_coherent(the_D)
        print(current_u)
        # the_K=np.linalg.diag

def get_coherent(the_D):
    coherent_max=float("-inf")
    for i in the_D.shape[0]:
        for j in the_D.shape[0]:
            if i==j:
                continue
            temp_coherent_max=(the_D[:,i]*the_D[:,j]).sum()
            if temp_coherent_max>coherent_max:
                coherent_max=temp_coherent_max
    return coherent_max

aa=np.array([[1,0],[0,2]])
bb=np.array([[3,4],[5,6]])
cc=np.dot(aa,bb)
pdb.set_trace()