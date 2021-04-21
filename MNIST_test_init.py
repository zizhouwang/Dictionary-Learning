# -*- coding: utf-8 -*-
"""
Created on April 29 2019

@author: Khanh-Hung TRAN
@work : CEA Saclay, France
@email : khanhhung92vt@gmail.com or khanh-hung.tran@cea.fr
"""

import pdb
import json
import numpy as np
from SSDL_GU import *
from sklearn.decomposition import SparseCoder
from numpy.linalg import norm
import sys
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
import time
from mnist import MNIST

def load_local_json_to_obj(path):
    str = open(path, encoding="utf-8").read()
    return np.array(json.loads(str))

t=time.time()

np.random.seed(int(t)%100)

n_classes = 10
data_count=10000

start_train_number=200
test_number=500
all_number=start_train_number+test_number

mndata = MNIST('samples')
images_test, labels_test = mndata.load_testing()
# imdb_data = np.load('mnist.npz')
# imdb_data_dict = dict(zip(('test_data', 'train_data', 'train_label', 'test_label'), (imdb_data[k] for k in imdb_data)))

# train_data, train_label, test_data, test_label = imdb_data_dict['train_data'], imdb_data_dict['train_label'], imdb_data_dict['test_data'], imdb_data_dict['test_label']
# data=train_data.reshape(data_count, 28*28).astype('float32')
data=np.array(images_test).astype('float32')
labels=np.array(labels_test)

""" Parameters in optimization  """
n_atoms = 200
n_neighbor = 8
lamda = 0.5
beta = 1.
gamma = 1.
mu = 2.*gamma
seed = 0
r = 2.
c = 1.

n_iter = 15
n_iter_sp = 50
n_iter_du = 50 

seed = 0 # to save the way initialize dictionary
n_iter_sp = 50 #number max of iteration in sparse coding
n_iter_du = 50 # number max of iteration in dictionary update
n_iter = 15 # number max of general iteration

D_all = load_local_json_to_obj('D_all_init_'+str(start_train_number)+'.txt')
W_all = load_local_json_to_obj('W_all_init_'+str(start_train_number)+'.txt')
A_all = load_local_json_to_obj('A_all_init_'+str(start_train_number)+'.txt')

indexs=np.array(np.where(labels==0))[0]
np.random.shuffle(indexs)
indexs=indexs[0:test_number]
Y_test=np.array(data[indexs],dtype=float).transpose()/255.
Y_test = preprocessing.normalize(Y_test.T, norm='l2').T*5
coder = SparseCoder(dictionary=D_all.T,transform_alpha=lamda/2., transform_algorithm='lasso_cd')
X_test=(coder.transform(Y_test.T)).T
the_H=np.dot(W_all,X_test)
right_num=0.
for i in range(test_number):
    pre=the_H[:,i].argmax()
    if pre==0:
        right_num+=1.
print('accuracy : '+str(right_num/test_number))

indexs=np.array(np.where(labels!=0))[0]
np.random.shuffle(indexs)
indexs=indexs[0:test_number*10]
Y_test=np.array(data[indexs],dtype=float).transpose()/255.
Y_test = preprocessing.normalize(Y_test.T, norm='l2').T*5
coder = SparseCoder(dictionary=D_all.T,transform_alpha=lamda/2., transform_algorithm='lasso_cd')
X_test=(coder.transform(Y_test.T)).T
the_H=np.dot(W_all,X_test)
right_num=0.
for i in range(test_number):
    pre=the_H[:,i].argmax()
    if pre!=0:
        right_num+=1.
print('accuracy : '+str(right_num/test_number))
