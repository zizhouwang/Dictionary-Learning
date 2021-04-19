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

t=time.time()

np.random.seed(int(t)%100)

n_classes = 10
data_count=60000

start_train_number=200
test_number=500
all_number=start_train_number+test_number
imdb_data = np.load('mnist.npz')
imdb_data_dict = dict(zip(('test_data', 'train_data', 'train_label', 'test_label'), (imdb_data[k] for k in imdb_data)))

train_data, train_label, test_data, test_label = imdb_data_dict['train_data'], imdb_data_dict['train_label'], imdb_data_dict['test_data'], imdb_data_dict['test_label']
data=train_data.reshape(data_count, 28*28).astype('float32')
labels=np.array(train_label)

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

def load_local_json_to_obj(path):
    str = open(path, encoding="utf-8").read()
    return np.array(json.loads(str))

def test(inputs):
    Y_test=inputs.reshape(784,1)
    """ To see accuracy for labelled samples and for unlabelled samples (in training phase) """
    #    X_labelled = X_all[:,:n_labelled]
    #    X_unlabelled = X_all[:,n_labelled:]
    #    
    #    n_correct_train_cl = np.array([y[i] == np.argmax(np.dot(W, X_labelled[:, i])+b)
    #                                      for i in range(X_labelled.shape[1])]).nonzero()[0].size/float(X_labelled.shape[1])
    #    
    #    n_correct_unlabelled_cl = np.array([y_unlabelled[i] == np.argmax(np.dot(W, X_unlabelled[:, i])+b)
    #                                      for i in range(X_unlabelled.shape[1])]).nonzero()[0].size/float(X_unlabelled.shape[1])

    """ Sparse coding for testing samples if beta != 0 (with manifold structure preservation) """

    n_test = np.shape(Y_test)[1]
    bb1 = np.zeros((n_atoms,n_test))
    neigh = NearestNeighbors(n_neighbors=n_neighbor+1)
    neigh.fit(Y_all.T)

    for i in range(n_test):
        a = inputs
        indice_test = neigh.kneighbors(np.atleast_2d(a), n_neighbor, return_distance=False)[0]
        weight_all = np.zeros(Y_all.shape[1])

        C = np.zeros((n_neighbor,n_neighbor))
        for k in range (n_neighbor):
            for l in range(k,n_neighbor):
                C[k,l] = np.sum((a - Y_all[:,indice_test[k]]) * (a - Y_all[:,indice_test[l]]))
                if k != l:
                    C[l,k] = C[k,l]
        if np.linalg.cond(C) > 1/sys.float_info.epsilon :
            C = C + (np.trace(C) * 0.1) * np.eye(n_neighbor)
            if np.linalg.cond(C) > 1/sys.float_info.epsilon :
                print("please check matrix C, not invertible")        
            
        invC = np.linalg.linalg.inv(C)
        denum = np.sum(invC)
        for k in range (n_neighbor):            
            weight_all[indice_test[k]] = np.sum(invC[k,:])/denum
        bb1[:,i] = np.dot(X_all,weight_all)

        
    _Y = np.vstack((Y_test,np.sqrt(beta)*bb1))
    _D = np.vstack((D, np.sqrt(beta) * np.eye(n_atoms)))
    coder = SparseCoder(dictionary=_D.T,transform_alpha=lamda/2., transform_algorithm='lasso_cd')
    # coder = SparseCoder(dictionary=D.T,transform_alpha=lamda/2., transform_algorithm='lasso_cd')
    X_test =coder.transform(_Y.T).T

    res=np.argmax(np.dot(W, X_test[:, i])+b)
        
    """ Results """ 

    print("res :" + str(res))   

# test_inputs=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 36.141176470588235, 162.63529411764705, 253.9921568627451, 253.9921568627451, 256.0, 253.9921568627451, 253.9921568627451, 253.9921568627451, 253.9921568627451, 167.65490196078431, 138.54117647058823, 28.109803921568627, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 39.15294117647059, 176.69019607843137, 246.96470588235294, 252.98823529411766, 252.98823529411766, 252.98823529411766, 253.9921568627451, 252.98823529411766, 252.98823529411766, 252.98823529411766, 252.98823529411766, 252.98823529411766, 252.98823529411766, 220.86274509803923, 18.070588235294117, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 62.24313725490196, 219.85882352941175, 252.98823529411766, 252.98823529411766, 252.98823529411766, 252.98823529411766, 209.81960784313725, 154.60392156862744, 125.49019607843137, 44.17254901960784, 149.5843137254902, 153.6, 231.90588235294118, 252.98823529411766, 252.98823529411766, 187.73333333333332, 13.050980392156863, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 19.07450980392157, 191.74901960784314, 252.98823529411766, 252.98823529411766, 242.94901960784313, 163.6392156862745, 39.15294117647059, 6.023529411764706, 0.0, 0.0, 0.0, 0.0, 0.0, 8.031372549019608, 63.247058823529414, 201.78823529411764, 246.96470588235294, 30.11764705882353, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.058823529411764, 219.85882352941175, 252.98823529411766, 252.98823529411766, 233.91372549019607, 55.21568627450981, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 73.28627450980392, 89.34901960784313, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.047058823529412, 179.7019607843137, 252.98823529411766, 252.98823529411766, 233.91372549019607, 59.23137254901961, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 79.30980392156863, 189.74117647058824, 89.34901960784313, 19.07450980392157, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 34.13333333333333, 252.98823529411766, 252.98823529411766, 251.98431372549018, 93.36470588235294, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 32.12549019607843, 221.86666666666667, 252.98823529411766, 252.98823529411766, 55.21568627450981, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 53.207843137254905, 252.98823529411766, 252.98823529411766, 163.6392156862745, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 67.26274509803922, 252.98823529411766, 252.98823529411766, 252.98823529411766, 55.21568627450981, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 143.5607843137255, 252.98823529411766, 252.98823529411766, 39.15294117647059, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 18.070588235294117, 187.73333333333332, 252.98823529411766, 252.98823529411766, 88.34509803921569, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 143.5607843137255, 252.98823529411766, 252.98823529411766, 10.03921568627451, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.3921568627451, 252.98823529411766, 252.98823529411766, 117.45882352941176, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 144.56470588235294, 253.9921568627451, 253.9921568627451, 10.03921568627451, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 243.95294117647057, 253.9921568627451, 55.21568627450981, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 115.45098039215686, 252.98823529411766, 252.98823529411766, 88.34509803921569, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 70.27450980392157, 249.9764705882353, 252.98823529411766, 55.21568627450981, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 34.13333333333333, 252.98823529411766, 252.98823529411766, 211.82745098039214, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.3921568627451, 252.98823529411766, 252.98823529411766, 55.21568627450981, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 16.062745098039215, 203.79607843137254, 252.98823529411766, 251.98431372549018, 141.5529411764706, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.3921568627451, 252.98823529411766, 252.98823529411766, 55.21568627450981, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 52.20392156862745, 243.95294117647057, 252.98823529411766, 209.81960784313725, 36.141176470588235, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 124.48627450980392, 252.98823529411766, 252.98823529411766, 55.21568627450981, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 171.6705882352941, 252.98823529411766, 252.98823529411766, 233.91372549019607, 17.066666666666666, 0.0, 0.0, 0.0, 0.0, 53.207843137254905, 243.95294117647057, 252.98823529411766, 233.91372549019607, 36.141176470588235, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.035294117647059, 185.72549019607843, 252.98823529411766, 252.98823529411766, 228.89411764705883, 54.21176470588235, 8.031372549019608, 0.0, 7.027450980392157, 170.66666666666666, 252.98823529411766, 252.98823529411766, 144.56470588235294, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.035294117647059, 180.7058823529412, 252.98823529411766, 252.98823529411766, 252.98823529411766, 222.87058823529412, 154.60392156862744, 210.8235294117647, 252.98823529411766, 252.98823529411766, 239.93725490196078, 34.13333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.058823529411764, 104.4078431372549, 246.96470588235294, 252.98823529411766, 252.98823529411766, 252.98823529411766, 253.9921568627451, 252.98823529411766, 252.98823529411766, 196.7686274509804, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 35.13725490196079, 142.55686274509804, 224.87843137254902, 252.98823529411766, 253.9921568627451, 175.68627450980392, 100.3921568627451, 16.062745098039215, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# test(np.array(test_inputs))


D_all = load_local_json_to_obj('D_all_'+str(start_train_number)+'.txt')
the_W = load_local_json_to_obj('the_W_'+str(start_train_number)+'.txt')
the_A = load_local_json_to_obj('the_A_'+str(start_train_number)+'.txt')

indexs=np.array(np.where(labels==0))[0]
np.random.shuffle(indexs)
indexs=indexs[0:test_number]
Y_test=np.array(data[indexs],dtype=float).transpose()/255.
Y_test = preprocessing.normalize(Y_test.T, norm='l2').T*5
coder = SparseCoder(dictionary=D_all.T,transform_alpha=lamda/2., transform_algorithm='lasso_cd')
X_test=(coder.transform(Y_test.T)).T
the_H=np.dot(the_W,X_test)
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
the_H=np.dot(the_W,X_test)
right_num=0.
for i in range(test_number):
    pre=the_H[:,i].argmax()
    if pre!=0:
        right_num+=1.
print('accuracy : '+str(right_num/test_number))
