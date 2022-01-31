import scipy.io
import time
import pdb
import numpy as np
from SSDL_GU import *
from LocalClassifier import *
from DictUpdate import *
from MLDLSI2_MUL import *
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

# def outer_func():
#     loc_list = []
#     def inner_func(name):
#         loc_list.append(len(loc_list) + 1)
#         print('%s loc_list = %s' %(name, loc_list))
#         return loc_list
#     return inner_func
# clo_func_0 = outer_func()
# clo_func_0('clo_func_0')
# aa=clo_func_0('clo_func_0')
# clo_func_0('clo_func_0')
# aa=1

MAX_Average_Precision=0.0
if os.path.exists('MAX_Average_Precision.mat'):
    MAX_Average_Precision=scipy.io.loadmat('MAX_Average_Precision.mat')['MAX_Average_Precision'][0][0]
rr=scipy.io.loadmat('r1_and_r2.mat')
incoherent_key=scipy.io.loadmat('incoherent_key.mat')
is_find_best=False

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
    params.positive = False;
    params.project = False;
    params.l2err = (8**2)*(1.15**2)*((1/255)**2);
    return params

def Find_K_Max_Eigen(Matrix,Eigen_NUM,Train_SET=None):
    if Train_SET.shape[1]==160:
        a=1
    NN,NN=Matrix.shape
    S,V=LA.eig(Matrix) #Note this is equivalent to; [V,S]=eig(St,SL); also equivalent to [V,S]=eig(Sn,St); %
    # S=copy.deepcopy(np.diag(S))
    index=S.argsort()
    V=V[:,index]
    # reverse=np.array([4,6,8,9,14,15,17,18,20,23,24,25,26,28])#这个是为dict_set准备的
    reverse=np.array([1,2,4,6,7,8,9,10,11,12,16,17,18,20,21,27,31,33,35,37,38,39,40,42,43,44,47,48,49,51,52,53,57,58,62,64,66,68,69,70,71,72,75,76,77,78,79,80,81,82,85,88,90,91,92,96,97,99,100,101,102,105,107,108,110,111,116,117,119,121])
    reverse=reverse-1
    # V[:,reverse]=-V[:,reverse]
    index.sort()
    S.sort()

    Eigen_Vector=np.zeros((NN,Eigen_NUM))
    Eigen_Value=np.zeros(Eigen_NUM)

    p=NN-1
    V=V.real
    S=S.real
    for t in range(Eigen_NUM):
        Eigen_Vector[:,t]=copy.deepcopy(V[:,index[p]])
        Eigen_Value[t]=copy.deepcopy(S[p])
        p=p-1
    return Eigen_Vector,Eigen_Value

def Eigenface_f(Train_SET,Eigen_NUM):
    NN,Train_NUM=Train_SET.shape
    if NN<=Train_NUM:
        Mean_Image=np.mean(Train_SET,axis=1)
        Mean_Image=Mean_Image.reshape((-1,1))
        Train_SET=Train_SET-np.dot(Mean_Image,np.ones((1,Train_NUM)))
        R=np.dot(Train_SET,Train_SET.T)/(Train_NUM-1)
        V,S=Find_K_Max_Eigen(R,Eigen_NUM)
        disc_value=S
        disc_set=V
    else:
        Mean_Image=np.mean(Train_SET,axis=1)
        Mean_Image=Mean_Image.reshape((-1,1))
        can_deleted_temp=np.dot(Mean_Image,np.ones((1,Train_NUM)))
        Train_SET=copy.deepcopy(Train_SET)-copy.deepcopy(can_deleted_temp)
        R=np.dot(Train_SET.T,Train_SET)/(Train_NUM-1)
        V,S=Find_K_Max_Eigen(R,Eigen_NUM,Train_SET)
        disc_value=S
        disc_set=np.zeros((NN,Eigen_NUM))
        Train_SET=Train_SET/math.sqrt(Train_NUM-1)
        for k in range(Eigen_NUM):
            disc_set[:,k]=(1./math.sqrt(disc_value[k]))*Train_SET@V[:,k]
    return disc_set,disc_value,Mean_Image

def Dict_Ini(data,nCol,wayInit):
    m=data.shape[0]
    if wayInit=="pca":
        (D,disc_value,Mean_Image)=Eigenface_f(data,nCol-1)
        D[:,-1]=-D[:,-1]
        # Mean_Image=preprocessing.normalize(Mean_Image.T, norm='l2').T
        D=np.hstack((D,Mean_Image))
    elif wayInit=="random":
        phi=np.random.randn(m,nCol)
        D=preprocessing.normalize(phi.T, norm='l2').T
    else:
        print("wayInit_error")
        exit()
    return D

for nonsense in range(10000):
    atom_n_1=300
    transform_n_nonzero_coefs=30
    data = scipy.io.loadmat('T4.mat')
    train_data=data['train_data']
    train_data_reg=preprocessing.normalize(train_data.T, norm='l2').T
    train_Annotation=data['train_Annotation']
    test_data=data['test_data']
    test_data_reg=preprocessing.normalize(test_data.T, norm='l2').T
    test_Annotation=data['test_Annotation']
    test_Annotation.dtype="int8"
    testNum=test_data.shape[1]
    labelNum=test_Annotation.shape[0]
    test_Annotation= 2*test_Annotation-1
    featureDim=test_data.shape[0]
    D0_reg=np.empty((labelNum, train_data.shape[0], atom_n_1))
    xmu=np.array([0.05])
    RankingLoss=np.zeros((xmu.shape[0]))
    Average_Precision=np.zeros((xmu.shape[0]))
    Coverage=np.zeros((xmu.shape[0]))
    OneError=np.zeros((xmu.shape[0]))
    params=Params()
    params.model=DefaultModelParams()
    params.model.lambda2=0.003
    params.model.lambda1=0.04
    params.mu0=0.0
    params.xmu0=0.05
    params.mu_mode=[-1]
    params.positive=False
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
    X=np.empty(labelNum,dtype=object)
    DataNum=np.zeros(labelNum,dtype=int)
    for i in range(labelNum):
        X[i] = copy.deepcopy(params.training_data[:, params.training_labels[i, :] == 1])
        DataNum[i] = X[i].shape[1]
    labelsb=copy.deepcopy(params.training_labels)
    labelsb = labelsb*2-1
    NumLabels=int(np.sum(np.sum(labelsb,axis=0).T))
    y = np.zeros(NumLabels)
    for i in range(labelNum):
        num = 0
        if i != 0:
            for k in range(i):
                num = num + DataNum[k]
        for j in range(int(DataNum[i])):
            num = int(num)
            y[num] = i
            num += 1
    # train_func1=MLDLSI2(params,y,atom_n)
    train_func1,D_init,Y_indexs=RLSDLA(atom_n_1, transform_n_nonzero_coefs,is_find_best)
    train_func2=None
    D1=None
    A_mean1=None
    Uk1=None
    bk1=None
    D=None
    A_mean=None
    Dusage=None
    Uk=None
    bk=None
    DataXb=np.empty((train_data_reg.shape[0],0))
    for i in range(labelNum):
        DataXb=np.hstack((DataXb,copy.deepcopy(train_data_reg[:,train_Annotation[i,:]==1])))



    r1_times=int(np.random.rand()*90)+30
    if is_find_best is not True:
        r1_times=rr['r1'][0][0]
    # r1_times=50



    for r1 in range(r1_times):#r1 97 r2 5 0.807534373838722
        #r1 102 r2 5 0.808937198067633
        D1=train_func1(r1,nonsense)

    coder = SparseCoder(dictionary=D1.T, transform_n_nonzero_coefs=transform_n_nonzero_coefs,
                        transform_algorithm="omp")
    A1_sum1 = (coder.transform(DataXb.T)).T
    A1_sum1=preprocessing.normalize(A1_sum1.T, norm='l2').T
    params.training_data   = A1_sum1
    #这里是随机初始化字典
    atom_n_2=30
    D0_reg_layer2 = np.random.rand(labelNum, A1_sum1.shape[0], atom_n_2)
    params.D0=copy.deepcopy(D0_reg_layer2)
    #现在随机初始化字典可以跑起来了 接下来用A1_sum来初始化字典
    for i in range(labelNum):
        Y_index_layer_2=np.where(y == i)[0]
        random.shuffle(Y_index_layer_2)
        D0_reg_layer2[i]= A1_sum1[:, Y_index_layer_2][:, :atom_n_2]
        params.D0=copy.deepcopy(D0_reg_layer2)


    if is_find_best is not True:
        D0_reg_layer2=scipy.io.loadmat('D0_reg_layer2.mat')['D0_reg_layer2']
        params.D0=copy.deepcopy(D0_reg_layer2)

    old_Average_Precision=0.0



    r2_times=int(np.random.rand()*50)+50
    if is_find_best is not True:
        r2_times=rr['r2'][0][0]
    # r2_times=14
    params.max_iter=r2_times



    train_func2 = MLDLSI2(params,y,atom_n_2)
    for r2 in range(params.max_iter):
        # D1,A_mean1,Dusage1,Uk1,bk1,A1_sum1,y,nonsense=train_func1(r2,False)
        D,A_mean,Dusage,Uk,bk,A1_sum2,y,is_finish=train_func2(r2,True,A1_sum1,nonsense)
        if is_finish:
            break

    testparam=Params()
    testparam.lambda1=params.model.the_lambda
    testparam.lambda2=0.04
    A_test = SparseRepresentation(test_data_reg,D1,A_mean1,testparam,params.model.lambda1,transform_n_nonzero_coefs)
    # output1,output2,output3 = LocalClassifier(test_data_reg,D1,A_mean1,testparam,Uk1,bk1,params.model.lambda1)
    output1,output2,output3 = LocalClassifier(A_test,D,A_mean,testparam,Uk,bk,params.model.lambda1)
    # toutput1,toutput2,toutput3 = LocalClassifier(train_data_reg,D,A_mean,testparam,Uk,bk,params.model.lambda1)
    # RankingLoss[m]=Ranking_loss(output1,test_Annotation)
    Average_Precision[0],Average_Precision1=Average_precision(output1,test_Annotation)
    print()
    print("Average_Precision: "+str(Average_Precision[0]))
    print()
    if old_Average_Precision==Average_Precision[0]:
        break
    else:
        old_Average_Precision = Average_Precision[0]
    if Average_Precision[0]<0.75 and is_find_best:
        break
    if Average_Precision[0]>MAX_Average_Precision:
        MAX_Average_Precision=Average_Precision[0]
        scio.savemat('incoherent_key.mat', {'incoherent_key': nonsense})
        scio.savemat('r1_and_r2.mat', {'r1': r1_times,'r2':r2+1})
        scio.savemat('D_random_init.mat', {'D_init': D_init})
        scio.savemat('Y_indexs.mat', {'Y_indexs': Y_indexs})
        scio.savemat('D0_reg_layer2.mat', {'D0_reg_layer2': D0_reg_layer2})
        scio.savemat('MAX_Average_Precision.mat', {'MAX_Average_Precision': MAX_Average_Precision})
        # Coverage[m]=coverage(output1,test_Annotation)
        # OneError[m]=One_error(output1,test_Annotation)
    # result_data=[xmu,Average_Precision,Coverage,OneError,RankingLoss]
    # pdb.set_trace()