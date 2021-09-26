import scipy.io
import time
import pdb
import numpy as np
from SSDL_GU import *
from LocalClassifier import *
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
import random
from numpy.matlib import repmat
import copy
import scipy.io as scio

class Params:
    pass

def DictUpdate(D0=None,AAt=None,XAt=None,D2=None,params=None):
    if D0 is None:
        params = Params()
        params.mu = 0
        params.xmu = 0
        params.max_iter = 1
        params.amuco = 0
        params.reg_delta = 1e-20
        params.armijo_a = 0.1
        params.armijo_b = 0.5
        params.armijo_m = 10
        params.armijo_s = 1
        params.scaling = 'diag'
        params.debug = 0
        params.debug_armijo = 0
        params.positive = False
        D = params
        return D
    D=copy.deepcopy(D0)
    M,K=D.shape
    I_M=np.eye(M)
    I_K=np.eye(K)
    if params.debug:
        print("eig AAt:min "+str(np.diag(AAt).min())+" max "+str(np.diag(AAt).max()))
    armijo_m_stats=np.zeros((K,1))
    stuck_stats=np.zeros((K,1))
    dCost=0
    if params.mu==0 and params.xmu==0:
        for J in range(params.max_iter):
            for k in range(K):
                g_k=2*(D@AAt[:,k]-XAt[:,k])
                D_k=D[:,k]-(0.5/max(AAt[k,k],params.reg_delta))*g_k
                if params.positive:
                    D_k[D_k<0]=0.
                if LA.norm(D_k)==0:
                    print("Ooops!")
                    D_k=np.random.randn(M,1)
                D[:,k]=(1/LA.norm(D_k))*D_k
    elif params.mu==0 and D2 is not None and D2.shape[0]>0:
        DDt2=D2@D2.T
        for J in range(params.max_iter):
            if params.debug:
                print(str(J)+"/"+str(params.max_iter))
            for k in range(K):
                D_k=copy.deepcopy(D0[:,k])
                g_k=2*(D0@AAt[:,k]-XAt[:,k])+(2*params.xmu)*(DDt2)@D_k
                if params.scaling=="full":
                    a=2*AAt[k,k]
                    H_k=a*I_M+(2*params.xmu)*DDt2
                    # L_k=modchol(H_k)
                    pass
                    # 懒得写了
                    print("error 153")
                    pdb.set_trace()
                elif params.scaling=="diag":
                    a=max(AAt[k,k],params.reg_delta)
                    H_k=2*a+(2*params.xmu)*np.diag(DDt2)
                    d_k=-g_k/H_k
                else:
                    d_k=-g_k
                    # 懒得写了
                    print("error 163")
                    pdb.set_trace()
                s=params.armijo_s
                acceptable_decrease=-params.armijo_a*g_k.T@d_k
                if acceptable_decrease<0:
                    print("Not a descent direction")
                    pdb.set_trace()
                D0=copy.deepcopy(D)
                D_k0=copy.deepcopy(D0[:,k])
                Ck=-XAt[:,k]+D0@AAt[:,k]
                Bk=D_k0.T@DDt2
                for j in range(params.armijo_m):
                    D_k=D_k0+s*d_k
                    if params.positive:
                        D_k[D_k<0]=0.
                    D_k=D_k*(1./LA.norm(D_k))
                    D[:,k]=copy.deepcopy(D_k)
                    dD_k=(D_k-D_k0)
                    df=-(2*Ck.T@dD_k+AAt[k,k]*np.sum(dD_k**2))
                    if params.debug_armijo:
                        print("df="+str(df)+" target="+str(acceptable_decrease))
                    if df>=acceptable_decrease:
                        break
                    s=s*params.armijo_b
                    acceptable_decrease=acceptable_decrease*params.armijo_b
                if df<0:
                    D[:,k]=copy.deepcopy(D_k0)
                    df=0
                dCost+=df
                if j>=params.armijo_m:
                    break
                    stuck_stats[j]=1
                armijo_m_stats[k]=j
                if params.debug:
                    print("\b\b\b\b\b\b\b\b\b")
            if params.debug:
                print("\r")
        if params.debug:
            print()
            print("Average armijo iterations:"+str(np.mean(armijo_m_stats)))
            print("Stuck:"+str(100*np.mean(stuck_stats)))
            print("dCost="+str(dCost))
    else:
        pass
        # 懒得写了
        print("error 173")
        pdb.set_trace()
    return D