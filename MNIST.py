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

np.random.seed(random_state)

n_classes = 10

""" Take mnist from keras """  
# from keras.datasets import mnist
# (data, labels), _ = mnist.load_data()
# data = data.reshape(60000, 28*28).astype('float32')
# labels = np.array(labels)


""" Or load mnist from downloaded data """  
#data = np.load("MNIST/data_training.npy")
#labels = np.load("MNIST/labels_training.npy")
#labels = np.array(labels)
imdb_data = np.load('mnist.npz')
# 'train_data', 'train_label', 'test_data', 'test_label'
imdb_data_dict = dict(zip(('test_data', 'train_data', 'train_label', 'test_label'), (imdb_data[k] for k in imdb_data)))

train_data, train_label, test_data, test_label = imdb_data_dict['train_data'], imdb_data_dict['train_label'], imdb_data_dict['test_data'], imdb_data_dict['test_label']
data=train_data.reshape(60000, 28*28).astype('float32')
labels=np.array(train_label)

def get_diff_and_degree_of_sparse(Y_all,coder,caled_number):
    start=0
    end=0
    # X_all=(coder.transform(Y_all.T)).T #X_all:2000x20001
    # Y_constructed_all=np.dot(D,X_all) #Y_constructed_all:784x20001
    # Y_diff_all=Y_all-Y_constructed_all
    Y_diff_num=np.zeros(10)
    X_nonzeros_num=np.zeros(10)
    for i in range(n_classes):
        print(i)
        sys.stdout.flush()
        end+=caled_number[i]
        X_part=(coder.transform(Y_all.T[start:end])).T
        Y_constructed_part=np.dot(D,X_part)
        Y_diff_part=Y_all[:,start:end]-Y_constructed_part
        # temp=Y_diff_all[:,start:end]
        Y_diff_num[i]=Y_diff_part.sum()
        # temp1=X_all[:,start:end]
        X_nonzeros_num[i]=X_part.nonzero()[0].shape[0]
        start+=caled_number[i]
    print(Y_diff_num)
    print(X_nonzeros_num)
    pdb.set_trace()

def get_part_data_for_observe():
    Y_part=[]
    indexs=list([])
    for i in range(n_classes):
        label_indexs_part=np.array(np.where(labels==i))[0]



""" Run 5 times with different random selections """
tab_unlabelled = np.zeros(1)
tab_test = np.zeros(1)

for random_state in range (1):
    zero_label_indexs=np.array(np.where(labels==0))[0]
    pdb.set_trace()
    # np.random.shuffle(zero_label_indexs)
    index = list([])
    start_train_number=2000
    start_test_number=300
    all_number=start_train_number+start_test_number
    #start
    for i in range (n_classes):
        num_i = 0
        j = 0
        while num_i < all_number:
            j = j + 1
            if labels[j] == i:
                index.append(j)
                num_i = num_i + 1
    #end 10个类别，每个类别有300个下标，数组index总共三千个下标
    """ random selection for training set (20 labelled samples, 80 unlabelled samples) and testing set (100 samples) """
    #start
    index = np.array(index)
    for i in range (n_classes):
        np.random.shuffle(index[all_number*i:all_number*i + all_number])
                
        
    index_l = list([])
    for i in range (n_classes):
        index_l.extend(index[all_number*i:all_number*i +start_train_number])    
    #end 10个类别，每个类别有200个下标，从每个类别的200个下标随机取20个下标，该下标用于labels
    y_labelled = labels[index_l]
    pdb.set_trace()
    n_labelled = len(y_labelled)
    Y_labelled = np.array(data[index_l],dtype = float).transpose()/255.
    index_u = list([])
    # for i in range (n_classes):
    #     index_u.extend(index[300*i + 20 :300*i + 20])    
    
    y_unlabelled = labels[index_u]
    n_unlabelled = len(y_unlabelled)
    Y_unlabelled = np.array(data[index_u],dtype = float).transpose()/255.
    
    index_t = list([])
    for i in range (n_classes):
        index_t.extend(index[all_number*i + start_train_number :all_number*i + start_train_number + start_test_number])    
    
    y_test = labels[index_t]
    n_test = len(y_test)
    Y_test = np.array(data[index_t],dtype = float).transpose()/255.
    Y_all = np.hstack((Y_labelled,Y_unlabelled))
    """ Preprocessing (make each sample have 5 times norm unity) """
    Y_all = preprocessing.normalize(Y_all.T, norm='l2').T*5
    Y_test = preprocessing.normalize(Y_test.T, norm='l2').T*5
    """ Parameters in optimization  """
    n_atoms = start_train_number
    n_neighbor = 8
    lamda = 0.5
    beta = 1.
    gamma = 1.
    mu = 2.*gamma
    r = 2.
    c = 1.
    
    seed = 0 # to save the way initialize dictionary
    n_iter_sp = 50 #number max of iteration in sparse coding
    n_iter_du = 50 # number max of iteration in dictionary update
    n_iter = 15 # number max of general iteration
    
    n_features = Y_all.shape[0]
    
    """ Contruct matrix for manifold structure preservation by LLE """
    # Lc = Construct_L_sparse_code(Y_all, n_neighbor)
    
    """ Start the process, initialize dictionary """
    # D = initialize_D(Y_all, n_atoms, y_labelled,n_labelled)
    D = initialize_single_D(Y_all, n_atoms, y_labelled,n_labelled,all_number,D_index=0)
    D = norm_cols_plus_petit_1(D,c)
    
    """ Label matrix for labelled samples """    
    # H = -np.ones((n_classes, n_labelled)).astype(float)
    # for i in range(n_labelled):
    #     H[int(y_labelled[i]), i] = 1
    
    """ Pseudo-Label matrix for unlabelled samples """
    # y_jck = -np.ones((n_classes, n_unlabelled, n_classes)).astype(float)
    # for k in range (n_classes):
    #     y_jck[k,:,k] = 1.
       
    """ Initialize Sparse code X_all """
    coder = SparseCoder(dictionary=D.T,transform_alpha=lamda/2., transform_algorithm='lasso_cd')
    X_single =(coder.transform(Y_all.T[0:start_train_number])).T #2000个列向量 每个列向量是一个图像的稀疏表征
    
    print("initializing classifier ... done")
    caled_number=np.zeros(n_classes,dtype=int)
    for i in range(n_classes):
        caled_number[i]=start_train_number
    for i in range(60000):
        for j in range(n_classes):
            print("start update")
            start=start_train_number*j+i*j
            end=start+start_train_number+i
            the_B=np.dot(Y_all[:,start:end],X_single.T)
            the_C=np.zeros((n_atoms,n_atoms))
            the_C=np.dot(X_single,X_single.T)
            np.random.shuffle(zero_label_indexs)
            new_index=[zero_label_indexs[0]]
            new_y=np.array(data[new_index],dtype = float).transpose()/255.
            new_y=preprocessing.normalize(new_y.T, norm='l2').T*5
            new_y.reshape(n_features,1)
            new_label=labels[new_index]
            new_x=(coder.transform(new_y.T)).T
            new_B=the_B+np.dot(new_y,new_x.T)
            new_C=np.array(the_C-(np.matrix(the_C)*np.matrix(new_x)*np.matrix(new_x.T)*np.matrix(the_C))/(np.matrix(new_x.T)*np.matrix(the_C)*np.matrix(new_x)+1)) #matrix inversion lemma(Woodbury matrix identity)
            new_D=np.dot(new_B,new_C)
            new_D = norm_cols_plus_petit_1(new_D,c)
            D=np.copy(new_D)
            coder = SparseCoder(dictionary=D.T,transform_alpha=lamda/2., transform_algorithm='lasso_cd',transform_max_iter=1000)
            Y_all=np.hstack((Y_all[:,0:end],new_y,Y_all[:,end:]))
            caled_number[j]+=1
            # aaa =(coder.transform(new_y.T)).T
            # bbb =(coder.transform(Y_all.T[42:44])).T
            # X_single =(coder.transform(Y_all.T[start:end+1])).T #X_single:2000x201
            # aaa =(coder.transform(Y_all.T)).T
            get_diff_and_degree_of_sparse(Y_all,coder,caled_number)
            pdb.set_trace()
            # print("D_sum:"+str(D.sum()))
            # print("X_all_sum:"+str(X_single.sum()))

    # with open('X_all.txt', mode='a+', encoding="utf-8") as w:
    #     w.write(json.dumps(X_all.tolist()))
    # with open('Y_all.txt', mode='a+', encoding="utf-8") as w:
    #     w.write(json.dumps(Y_all.tolist()))
    # with open('D.txt', mode='a+', encoding="utf-8") as w:
    #     w.write(json.dumps(D.tolist()))
    # with open('W.txt', mode='a+', encoding="utf-8") as w:
    #     w.write(json.dumps(W.tolist()))
    # with open('b.txt', mode='a+', encoding="utf-8") as w:
    #     w.write(json.dumps(b.tolist())) 