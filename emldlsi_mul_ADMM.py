import scipy.io
import time
import pdb
import numpy as np
from SSDL_GU import *
from LocalClassifier import *
from DictUpdate import *
from MLDLSI2 import *
from learning_incoherent_dictionary import *
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

def cal_S(D,tao,P,X):
    return np.linalg.inv(D.T@D+tao*np.eye(D.shape[1]))@(tao*P@X+D.T@X)#公式4

def cal_P(S,tao,lamda,gamma,P,X,X_ba):
    return tao*S@X.T@(tao*X@X.T+lamda*X_ba@X_ba.T+gamma*np.eye(X.shape[0]))#公式5

def cal_D_T_U_rou(rou,rou_rate,D,T,U,X,S,beta,D_all,i):
    Djt_Dj=np.zeros((D.shape[1],D.shape[1]))
    for j in range(D_all.shape[0]):
        if j==i:
            continue
        Djt_Dj=Djt_Dj+D_all[j].T@D_all[j]
    D_new=(rou*(T-U)+X@S.T)@np.linalg.inv((rou*np.eye(S.shape[0])+S@S.T+beta*Djt_Dj))#公式10
    diff=np.sum(abs(D-D_new))
    D=D_new
    T=D+U#公式11
    T=preprocessing.normalize(T.T, norm='l2').T
    U=U+D-T#公式12
    # rou=rou*rou_rate
    return rou,D,T,U,diff

atom_n=30
transform_n_nonzero_coefs=30
data = scipy.io.loadmat('T4.mat') # 读取mat文件
D_init = scipy.io.loadmat('D_init.mat')['D0_reg'][0] # 读取mat文件
train_data=data['train_data']
train_data_reg=preprocessing.normalize(train_data.T, norm='l2').T
train_Annotation=data['train_Annotation']
train_Annotation=train_Annotation.astype(int)
test_data=data['test_data']
test_data_reg=preprocessing.normalize(test_data.T, norm='l2').T
test_Annotation=data['test_Annotation']
test_Annotation=test_Annotation.astype(int)
testNum=test_data.shape[1]
labelNum=test_Annotation.shape[0]
featureDim=test_data.shape[0]
atomNum=[atom_n,atom_n,atom_n,atom_n,atom_n]
one_label_data=np.empty(5,dtype=object)
one_label_data_all=np.zeros((train_data.shape[0],0))
for i in range(labelNum):
    one_label_data[i]=train_data[:,np.where(train_Annotation[i]==1)[0]]
    one_label_data_all=np.hstack((one_label_data_all,one_label_data[i]))
y = np.zeros(one_label_data_all.shape[1])
start=0
for i in range(labelNum):
    end=start+one_label_data[i].shape[1]
    y[start:end]=i
    start=end
lamda=0.005
tao=1
beta=0.08
gamma=1e-4
rou=1
rou_rate=1.2
# D=np.empty((labelNum,train_data.shape[0],atom_n))
D=np.random.randn(labelNum,train_data.shape[0],atom_n)
P=np.random.randn(labelNum,atom_n,train_data.shape[0])
U=np.random.randn(labelNum,train_data.shape[0],atom_n)
T=copy.deepcopy(D)
for l in range(10):
    a=1
    for i in range(labelNum):
        S_i=cal_S(D[i],tao,P[i],one_label_data[i])
        X_ba=one_label_data_all[:,np.where(y!=i)[0]]
        P[i]=cal_P(S_i,tao,lamda,gamma,P[i],one_label_data[i],X_ba)
        converge_time=0
        for nonsense in range(1000):
            rou,D[i],T[i],U[i],diff=cal_D_T_U_rou(rou, rou_rate, D[i], T[i], U[i], one_label_data[i], S_i, beta, D, i)
            if diff<1e-12:
                converge_time+=1
            else:
                converge_time=0
            if converge_time>5:
                break
scio.savemat('D_and_P.mat', {'D': D,'P':P})
output=np.empty((labelNum,test_data_reg.shape[1]))
for i in range(labelNum):
    output[i]=1./np.sum(abs(test_data_reg-D[i]@P[i]@test_data_reg),axis=0)#开始预测
Average_Precision,Average_Precision1=Average_precision(output,test_Annotation)#计算平均精度
print(Average_Precision)
aa=1