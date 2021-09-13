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

class Params:
    pass
# params=Params()
# params.model=Params()
# params.model.lambda2=0.003
# params.model.lambda1=0.04
# params.mu0=0.0
# params.xmu0=0.05
# params.mu_mode=[-1]
# pdb.set_trace()
# for i in range(10,1,-1):
#     print(i)
pdb.set_trace()

def DefaultModelParams():
    params=Params()
    params.reg_mode = 2; 
    params.the_lambda = 0.01;
    params.theta = 30.0;
    params.kappa = 2.5;
    params.beta = 0.05;
    params.reg_type = 'l1';
    params.lambda_min = 10;
    params.lambda_max = 150;
    params.lla_iter = 5;
    params.L = 0; 
    params.positive = false;
    params.project = false;
    params.l2err = 8^2*1.15^2*(1/255)^2;
    return params

def Find_K_Max_Eigen(Matrix,Eigen_NUM):

    NN,NN=Matrix.shape
    V,S=LA.eig(Matrix) #Note this is equivalent to; [V,S]=eig(St,SL); also equivalent to [V,S]=eig(Sn,St); %

    S=np.diag(S)
    index=S.argsort()
    S=S.sort()

    Eigen_Vector=zeros((NN,Eigen_NUM));
    Eigen_Value=zeros(Eigen_NUM);

    p=NN
    for t in Eigen_NUM:
        Eigen_Vector[:,t]=V[:,index[p]]
        Eigen_Value[t]=S[p]
        p=p-1
    return Eigen_Vector,Eigen_Value

def Eigenface_f(Train_SET,Eigen_NUM):
    NN,Train_NUM=Train_SET.shape
    if NN<=Train_NUM:
        Mean_Image=np.mean(Train_SET,axis=1)
        Train_SET=Train_SET-np.dot(Mean_Image,np.ones((1,Train_NUM)))
        R=np.dot(Train_SET,Train_SET.T)/(Train_NUM-1)
        V,S=Find_K_Max_Eigen(R,Eigen_NUM)
        disc_value=S
        disc_set=V
    else:
        Mean_Image=np.mean(Train_SET,axis=1)
        Train_SET=Train_SET-np.dot(Mean_Image,np.ones((1,Train_NUM)))
        R=np.dot(Train_SET.T,Train_SET)/(Train_NUM-1)
        V,S=Find_K_Max_Eigen(R,Eigen_NUM)
        disc_value=S
        disc_set=np.zeros((NN,Eigen_NUM))
        Train_SET=Train_SET/math.sqrt(Train_NUM-1)
        for k in range(Eigen_NUM):
            disc_set[:,k]=(1./math.sqrt(disc_value[k]))*Train_SET*V[:,k]

def Dict_Ini(data,nCol,wayInit):
    m=data.shape[0]
    if wayInit=="pca":
        (D,disc_value,Mean_Image)=Eigenface_f(data,nCol-1)
        Mean_Image=preprocessing.normalize(Mean_Image.T, norm='l2').T
        D=np.hstack((D,Mean_Image))
    elif wayInit=="random":
        phi=np.random.randn(m,nCol)
        D=preprocessing.normalize(phi.T, norm='l2').T
    else:
        print("wayInit_error")
        exit()

# data = scipy.io.loadmat('clothes5.mat') # 读取mat文件
data = scipy.io.loadmat('T4.mat') # 读取mat文件
# print(data.keys())  # 查看mat文件中的所有变量
train_data=data['train_data']
train_data_reg=preprocessing.normalize(train_data.T, norm='l2').T
aa=np.stack((train_data,train_data),axis=1)
train_Annotation=data['train_Annotation']
test_data=data['test_data']
test_data_reg=preprocessing.normalize(test_data.T, norm='l2').T
test_Annotation=data['test_Annotation']
testNum=test_data.shape[1]
labelNum=test_Annotation.shape[0]
featureDim=test_data.shape[0]
atom_n=30
atomNum=[atom_n,atom_n,atom_n,atom_n,atom_n]
D0=np.empty((labelNum,train_data.shape[0],atom_n))
D0_reg=np.empty((labelNum,train_data.shape[0],atom_n))
# Dic_reg_para=np.empty((labelNum,atom_n))
xmu=np.array([0.05])
RankingLoss=np.zeros((xmu.shape[0]))
Average_Precision=np.zeros((xmu.shape[0]))
Coverage=np.zeros((xmu.shape[0]))
OneError=np.zeros((xmu.shape[0]))
for m in range(xmu.shape[0]):
    for i in range(labelNum):
        cdat=train_data_reg[:,train_Annotation[i,:]==1]
        nRow,nCol=cdat.shape
        if atomNum[i]>min(featureDim,cdat.shape[1]):
            wayInit1="pca"
            wayInit2="random"
            atomNum1=min(featureDim,cdat.shape[1])
            atomNum2=atomNum[i]-atomNum1
            dict1=Dict_Ini(cdat,atomNum1,wayInit1)
            dict2=Dict_Ini(cdat,atomNum2,wayInit2)
            the_dict=np.hstack((dict1,dict2))
        else:
            wayInit="pca"
            the_dict=Dict_Ini(cdat,atomNum[i],wayInit)
        D0[i]=the_dict
        D0_reg[i]=preprocessing.normalize(the_dict.T, norm='l2').T
    params=DefaultModelParams()
    params.model=Params()
    params.model.lambda2=0.003
    params.model.lambda1=0.04
    params.mu0=0.0
    params.xmu0=0.05
    params.mu_mode=[-1]
    params.positive=False
    params.max_iter=1000
    params.min_change=1e-5
    params.batch_size=0
    params.test_size=0
    params.resume=False
    params.do_control_class   = False
    params.xval_step = 10
    params.remember_factor = 0.4
    params.output_dir      = 'results/dict'
    params.training_data   = train_data_reg
    params.testing_data    = []
    params.training_labels = train_Annotation
    params.testing_labels  = []
    params.update_method   = 'pg'
    params.debug           = 0
    params.base_name       = 'global'
    params.discard_unused_atoms    = 0.005
    params.discard_constant_patches = 0.001
    params.dict_update     = DictUpdate()
    params.dict_update.xcorr = 1
    params.D0 = D0_reg
    D,A_mean,Dusage,Uk,bk        = MLDLSI2(params)
    testparam=Params()
    testparam.lambda1=params.model.the_lambda
    testparam.lambda2=0.04
    output1,output2,output3 = LocalClassifier(test_data_reg,D,A_mean,testparam,Uk,bk,params.model.lambda1)
    toutput1,toutput2,toutput3 = LocalClassifier(train_data_reg,D,A_mean,testparam,Uk,bk,params.model.lambda1)
    test_Annotation= 2*test_Annotation-1
    RankingLoss[m]=Ranking_loss(output1,test_Annotation)
    Average_Precision[m]=Average_precision(output1,test_Annotation)
    Coverage[m]=coverage(output1,test_Annotation)
    OneError[m]=One_error(output1,test_Annotation)
result_data=[xmu,Average_Precision,Coverage,OneError,RankingLoss]
pdb.set_trace()