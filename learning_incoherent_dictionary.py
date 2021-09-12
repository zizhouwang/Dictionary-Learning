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
    gram=np.dot(the_D.T,the_D)
    for i in range(gram.shape[0]):
        gram[i][i]=-gram[i][i]
    current_u=gram.max()
    for i in range(gram.shape[0]):
        gram[i][i]=-gram[i][i]
    u0=current_u-(1-current_u)/2.
    print("u0:"+str(u0))
    print()
    print("current_u:"+str(current_u))
    print()
    while iIter<nIter and current_u>u0:
        the_K=copy.deepcopy(gram)
        the_K[the_K<-u0]=-u0
        the_K[the_K>u0]=u0
        for i in range(the_K.shape[0]):
            the_K[i][i]=1.
        gram=the_K
        eigenvalues,Q=LA.eig(gram)
        eigenvalues.dtype='float'
        Q.dtype='float'
        eigenvalues=np.diag(eigenvalues)
        eigenvalues[eigenvalues<0]=0
        part_D=np.dot(np.sqrt(eigenvalues),Q.T)
        for i in range(the_D.shape[0]):
            the_D[i,:]=part_D[i%part_D.shape[0],:]
        the_D=preprocessing.normalize(the_D.T, norm='l2').T
        the_C=np.dot(the_Y,np.dot(the_D,the_X).T)
        the_u,the_s,the_vt=LA.svd(the_C)
        the_W=np.dot(the_vt.T,the_u.T)
        the_D=np.dot(the_W,the_D)
        gram=np.dot(the_D.T,the_D)
        for i in range(gram.shape[0]):
            gram[i][i]=-gram[i][i]
        current_u=gram.max()
        for i in range(gram.shape[0]):
            gram[i][i]=-gram[i][i]
        print("u0:"+str(u0))
        print()
        print("current_u:"+str(current_u))
        print()
        sys.stdout.flush()
        iIter+=1
        # the_K=np.linalg.diag
    return the_D

def get_coherent(the_D):
    coherent_max=float("-inf")
    for i in range(the_D.shape[0]):
        for j in range(the_D.shape[0]):
            if i==j:
                continue
            temp_coherent_max=(the_D[:,i]*the_D[:,j]).sum()
            if temp_coherent_max>coherent_max:
                coherent_max=temp_coherent_max
    return coherent_max