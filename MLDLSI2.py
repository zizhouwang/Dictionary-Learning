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
from scipy.spatial.distance import pdist, squareform
import copy
import scipy.io as scio

def GetPrefix(params):
    path = 'C:/EMLDLSI/model/'
    prefix = path+'model-'+time.strftime('%Y-%m-%d-%H-%M',time.localtime(time.time()))


def MLDLSI2(params):#[D,A1_mean,Dusage,Uk,bk]
    NC=params.D0.shape[0]
    K=np.zeros(NC)
    labelname=np.arange(params.training_labels.shape[0])
    labelCorr=np.ones((NC,NC))
    if params.dict_update.xcorr==1:
        for i in NC:
            for j in NC:
                labelCorr[i,j]=pdist([params.training_labels[i,:]+0.,params.training_labels[j,:]+0.],'cosine')
    for i in NC:
        M,K[i]=params.D0[i].shape
    Xb=params.training_data
    labelsb=params.training_labels
    D=params.D0
    r0=1
    dDn=np.zeros(NC)
    finished=np.zeros(NC)
    AAt=np.empty(NC,dtype=object)
    A1=np.empty(NC,dtype=object)
    A_ini=np.empty(NC,dtype=object)
    XAt=np.empty(NC,dtype=object)
    Dusage=np.empty(NC,dtype=object)
    Dmean=np.empty(NC,dtype=object)
    Dvar=np.empty(NC,dtype=object)
    kt_usage=np.empty(NC,dtype=object)
    xc=np.empty(NC,dtype=object)
    A1_mean=np.empty(NC,dtype=object)

    for i in NC:
        Dusage[i]=np.zeros(K[i])
        Dmean[i]=np.zeros(K[i])
        Dvar[i]=np.zeros(K[i])
        AAt[i]=np.zeros((K[i],K[i]))
        XAt[i]=np.zeros((M,K[i]))
        A1_mean[i]=np.zeros((K[i],1))
    cost=np.zeros(params.max_iter,NC)
    l1_energy=np.zeros(params.max_iter,NC)
    l1_energy2=np.zeros(params.max_iter,NC)
    new_energy=np.zeros(params.max_iter,NC)
    mean_coherence=np.zeros(params.max_iter,NC)
    max_coherence=np.zeros(params.max_iter,NC)
    max_cross_coherence=np.zeros(params.max_iter,NC)
    mean_cross_coherence=np.zeros(params.max_iter,NC)
    accumulated_N=np.zeros(1,NC)
    acciter=np.zeros(params.max_iter,NC)

    prefix=GetPrefix(params)
    params2=copy.deepcopy(params)
    params.testing_data=[]
    params.training_data=[]
    params.testing_labels=[]
    params.training_labels=[]
    params.D0=[]
    scio.savemat(prefix+'-params.mat', {'params':params})
    params=params2