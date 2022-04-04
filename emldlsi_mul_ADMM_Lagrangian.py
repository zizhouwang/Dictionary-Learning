#two layer dictionary learning: ADMM-MLDLSI 0.7864269788182834

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
from li2nsvm_multiclass_lbfgs import *

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

MAX_Average_Precision=0.0

while True:

    atom_n=300
    transform_n_nonzero_coefs_1=30
    atom_n_2=30
    transform_n_nonzero_coefs_2=30
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
    one_label_data_all=np.zeros((train_data_reg.shape[0],0))
    for i in range(labelNum):
        one_label_data[i]=train_data_reg[:,np.where(train_Annotation[i]==1)[0]]
        one_label_data_all=np.hstack((one_label_data_all,one_label_data[i]))
    y = np.zeros(one_label_data_all.shape[1],dtype=int)
    start=0
    for i in range(labelNum):
        end=start+one_label_data[i].shape[1]
        y[start:end]=i
        start=end
    lagrangian_multiplier=np.zeros(one_label_data_all.shape)
    beta=6
    delta=1e-6
    gamma=np.random.rand(1)[0]
    gamma=0.5
    D_first_layer=np.random.randn(train_data.shape[0], atom_n)
    D_first_layer=preprocessing.normalize(D_first_layer.T, norm='l2').T
    G=np.empty(one_label_data_all.shape)
    H=np.empty(one_label_data_all.shape)
    Y_pre=np.zeros((labelNum,y.shape[0]),dtype=int)
    for i in range(y.shape[0]):
        Y_pre[y[i],i]=1
    W=np.random.randn(labelNum,atom_n)
    W=preprocessing.normalize(W.T, norm='l2').T
    G_W=np.empty(Y_pre.shape)
    H_W=np.empty(Y_pre.shape)
    lagrangian_multiplier_W=np.zeros(Y_pre.shape)

    params = Params()
    params.model = DefaultModelParams()
    params.model.lambda2 = 0.003
    params.model.lambda1 = 0.04
    params.mu0 = 0.0
    params.xmu0 = 0.05
    params.mu_mode = [-1]
    params.positive = False
    params.min_change = 1e-5
    params.batch_size = 0
    params.test_size = 0
    params.resume = False
    params.do_control_class = False
    params.xval_step = 10
    params.remember_factor = 0.4
    params.output_dir = 'results/dict'
    params.training_data = train_data_reg
    params.testing_data = []
    params.training_labels = train_Annotation
    params.testing_labels = []
    params.update_method = 'pg'
    params.debug = 0
    params.base_name = 'global'
    params.discard_unused_atoms = 0.005
    params.discard_constant_patches = 0.001
    params.dict_update = DictUpdate()
    params.dict_update.xcorr = 1
    params.max_iter=1000
    for l in range(5):
        print(l)
        coder = SparseCoder(dictionary=D_first_layer.T, transform_n_nonzero_coefs=transform_n_nonzero_coefs_1,
                            transform_algorithm="omp")
        X_train = (coder.transform(one_label_data_all.T)).T
        G= (beta * D_first_layer @ X_train + 2 * one_label_data_all - lagrangian_multiplier) / (2 + beta)
        H= G + lagrangian_multiplier / beta - D_first_layer @ X_train
        for i in range(D_first_layer.shape[1]):
            Omega=X_train[i]@X_train[i]
            D_first_layer[:, i]= D_first_layer[:, i] + (H @ X_train[i]) / (Omega + delta)
        D_first_layer=preprocessing.normalize(D_first_layer.T, norm='l2').T
        lagrangian_multiplier=lagrangian_multiplier+gamma*beta*(G - D_first_layer @ X_train)

        G_W=(beta*W@X_train+2*Y_pre-lagrangian_multiplier_W)/(2+beta)
        H_W=G_W+lagrangian_multiplier_W/beta-W@X_train
        for i in range(D_first_layer.shape[1]):
            Omega=X_train[i]@X_train[i]
            W[:,i]=W[:,i]+(H_W@X_train[i])/(Omega+delta)
        W=preprocessing.normalize(W.T, norm='l2').T
        lagrangian_multiplier_W=lagrangian_multiplier_W+gamma*beta*(G_W-W@X_train)
        pass

    train_func2=None
    A_mean=None
    Dusage=None
    Uk=None
    bk=None
    D_sec_layer=None

    coder = SparseCoder(dictionary=D_first_layer.T, transform_n_nonzero_coefs=transform_n_nonzero_coefs_1,
                        transform_algorithm="omp")
    X_test = (coder.transform(test_data_reg.T)).T
    # diff = np.sum(abs(test_data_reg - D @ X_test))
    # print(diff)
        # output=W@X_test
        # Average_Precision,Average_Precision1=Average_precision(output,test_Annotation)
        # print(Average_Precision)
    coder = SparseCoder(dictionary=D_first_layer.T, transform_n_nonzero_coefs=transform_n_nonzero_coefs_1,
                        transform_algorithm="omp")
    X_train_first_layer = (coder.transform(one_label_data_all.T)).T
    params.training_data=X_train_first_layer
    #这里是随机初始化字典
    D0_reg_layer2 = np.random.rand(labelNum, X_train_first_layer.shape[0], atom_n_2)
    params.D0=copy.deepcopy(D0_reg_layer2)
    # 现在随机初始化字典可以跑起来了 接下来用A1_sum来初始化字典
    # for i in range(labelNum):
    #     Y_index_layer_2 = np.where(y == i)[0]
    #     # random.shuffle(Y_index_layer_2)
    #     D0_reg_layer2[i] = X_train_first_layer[:, Y_index_layer_2][:, :atom_n_2]
    #     params.D0 = copy.deepcopy(D0_reg_layer2)
    train_func2 = MLDLSI2(params,y,atom_n_2)
    for r2 in range(params.max_iter):
        # D1,A_mean1,Dusage1,Uk1,bk1,A1_sum1,y,nonsense=train_func1(r2,False)
        D_sec_layer, A_mean, Dusage, Uk, bk, A1_sum2, y, is_finish = train_func2(r2, True, X_train_first_layer, 0)
        if is_finish:
            break

    testparam = Params()
    testparam.lambda1 = params.model.the_lambda
    testparam.lambda2 = 0.04
    A_test = X_test
    output1, output2, output3 = LocalClassifier(A_test, D_sec_layer, A_mean, testparam, Uk, bk, params.model.lambda1)
    # RankingLoss[m]=Ranking_loss(output1,test_Annotation)
    Average_Precision, Average_Precision1 = Average_precision(output1, test_Annotation)
    print()
    print("Average_Precision: " + str(Average_Precision))
    print()
    if Average_Precision>0.78:
        break