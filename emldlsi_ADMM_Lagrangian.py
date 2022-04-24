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
from li2nsvm_multiclass_lbfgs import *

atom_n=300
atom_n2=300
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
D=np.random.randn(one_label_data_all.shape[0],atom_n)
D=preprocessing.normalize(D.T, norm='l2').T
G=np.empty(one_label_data_all.shape)
H=np.empty(one_label_data_all.shape)
for l in range(5):
    print(l)
    coder = SparseCoder(dictionary=D.T, transform_n_nonzero_coefs=transform_n_nonzero_coefs,
                        transform_algorithm="omp")
    X_train = (coder.transform(one_label_data_all.T)).T
    G=(beta*D@X_train+2*one_label_data_all-lagrangian_multiplier)/(2+beta)
    H=G+lagrangian_multiplier/beta-D@X_train
    for i in range(D.shape[1]):
        Omega=X_train[i]@X_train[i]
        D[:,i]=D[:,i]+(H@X_train[i])/(Omega+delta)
    D=preprocessing.normalize(D.T, norm='l2').T
    lagrangian_multiplier=lagrangian_multiplier+gamma*beta*(G-D@X_train)
    pass

coder = SparseCoder(dictionary=D.T, transform_n_nonzero_coefs=transform_n_nonzero_coefs,
                    transform_algorithm="omp")
X_train = (coder.transform(one_label_data_all.T)).T
X_test_layer1 = (coder.transform(test_data_reg.T)).T

train_data_reg_layer2=copy.deepcopy(preprocessing.normalize(X_train.T, norm='l2').T)
D2=np.random.randn(train_data_reg_layer2.shape[0],atom_n2)
D2=preprocessing.normalize(D2.T, norm='l2').T
G2=np.empty(train_data_reg_layer2.shape)
H2=np.empty(train_data_reg_layer2.shape)
lagrangian_multiplier2=np.zeros(train_data_reg_layer2.shape)
Y_pre=np.zeros((labelNum,y.shape[0]),dtype=int)
for i in range(y.shape[0]):
    Y_pre[y[i],i]=1
W=np.random.randn(labelNum,atom_n2)
W=preprocessing.normalize(W.T, norm='l2').T
G_W=np.empty(Y_pre.shape)
H_W=np.empty(Y_pre.shape)
lagrangian_multiplier_W=np.zeros(Y_pre.shape)
for l in range(50):
    print(l)
    coder2 = SparseCoder(dictionary=D2.T, transform_n_nonzero_coefs=transform_n_nonzero_coefs,
                        transform_algorithm="omp")
    X_train = (coder2.transform(train_data_reg_layer2.T)).T
    G2=(beta*D2@X_train+2*train_data_reg_layer2-lagrangian_multiplier2)/(2+beta)
    H2=G2+lagrangian_multiplier2/beta-D2@X_train
    for i in range(D2.shape[1]):
        Omega=X_train[i]@X_train[i]
        D2[:,i]=D2[:,i]+(H2@X_train[i])/(Omega+delta)
    D2=preprocessing.normalize(D2.T, norm='l2').T
    lagrangian_multiplier2=lagrangian_multiplier2+gamma*beta*(G2-D2@X_train)

    G_W=(beta*W@X_train+2*Y_pre-lagrangian_multiplier_W)/(2+beta)
    H_W=G_W+lagrangian_multiplier_W/beta-W@X_train
    for i in range(D2.shape[1]):
        Omega=X_train[i]@X_train[i]
        W[:,i]=W[:,i]+(H_W@X_train[i])/(Omega+delta)
    # W=preprocessing.normalize(W.T, norm='l2').T
    lagrangian_multiplier_W=lagrangian_multiplier_W+gamma*beta*(G_W-W@X_train)

    coder2 = SparseCoder(dictionary=D2.T, transform_n_nonzero_coefs=transform_n_nonzero_coefs,
                        transform_algorithm="omp")
    X_test = (coder2.transform(X_test_layer1.T)).T
    output=W@X_test
    Average_Precision,Average_Precision1=Average_precision(output,test_Annotation)
    print(Average_Precision)
    pass
W=Y_pre@X_train.T@LA.inv(X_train@X_train.T)
