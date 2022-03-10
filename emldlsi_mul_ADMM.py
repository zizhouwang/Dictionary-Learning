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

atom_n=30
transform_n_nonzero_coefs=30
data = scipy.io.loadmat('T4.mat') # 读取mat文件
D_init = scipy.io.loadmat('D_init.mat')['D0_reg'][0] # 读取mat文件
train_data=data['train_data']
train_data_reg=preprocessing.normalize(train_data.T, norm='l2').T
train_Annotation=data['train_Annotation']
test_data=data['test_data']
test_data_reg=preprocessing.normalize(test_data.T, norm='l2').T
test_Annotation=data['test_Annotation']
test_Annotation.dtype="int8"
testNum=test_data.shape[1]
labelNum=test_Annotation.shape[0]
featureDim=test_data.shape[0]
atomNum=[atom_n,atom_n,atom_n,atom_n,atom_n]
D=np.empty((labelNum,train_data.shape[0],atom_n))
# D_reg=np.random.randn(labelNum,train_data.shape[0],atom_n)