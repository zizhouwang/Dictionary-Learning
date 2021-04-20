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

# mndata = MNIST('samples')

# images_train, labels_train = mndata.load_training()
# or
# images_test, labels_test = mndata.load_testing()

t=time.time()

np.random.seed(int(t)%100)

n_classes = 10
data_count=60000

start_train_number=200
test_number=0
all_number=start_train_number+test_number
imdb_data = np.load('mnist.npz')
imdb_data_dict = dict(zip(('test_data', 'train_data', 'train_label', 'test_label'), (imdb_data[k] for k in imdb_data)))

train_data, train_label, test_data, test_label = imdb_data_dict['train_data'], imdb_data_dict['train_label'], imdb_data_dict['test_data'], imdb_data_dict['test_label']
data=train_data.reshape(data_count, 28*28).astype('float32')
labels=np.array(train_label)

def load_local_json_to_obj(path):
    str = open(path, encoding="utf-8").read()
    return np.array(json.loads(str))

def get_diff_and_degree_of_sparse(Y_all,coder,caled_number):
    start=0
    end=0
    # X_all=(coder.transform(Y_all.T)).T #X_all:2000x20001
    # Y_constructed_all=np.dot(D,X_all) #Y_constructed_all:784x20001
    # Y_diff_all=Y_all-Y_constructed_all
    Y_diff_num=np.zeros(10)
    X_nonzeros_num=np.zeros(10)
    for i in range(n_classes):
        # print(i)
        end+=caled_number[i]
        X_part=(coder.transform(Y_all.T[start:end])).T
        Y_constructed_part=np.dot(D,X_part)
        Y_diff_part=Y_all[:,start:end]-Y_constructed_part
        # temp=Y_diff_all[:,start:end]
        Y_diff_num[i]=abs(Y_diff_part).sum()
        # temp1=X_all[:,start:end]
        X_nonzeros_num[i]=X_part.nonzero()[0].shape[0]
        start+=caled_number[i]
    print(Y_diff_num)
    print(X_nonzeros_num)
    sys.stdout.flush()
    # pdb.set_trace()

def get_part_data_and_observe(coder):
    Y_part=[]
    indexs=list([])
    caled_number=np.zeros(n_classes,dtype=int)
    one_class_number=300
    for i in range(n_classes):
        caled_number[i]=one_class_number
        label_indexs_part=np.array(np.where(labels==i))[0]
        np.random.shuffle(label_indexs_part)
        indexs.extend(label_indexs_part[0:one_class_number])
    Y_part=np.array(data[indexs],dtype=float).transpose()/255.
    Y_part = preprocessing.normalize(Y_part.T, norm='l2').T*5
    get_diff_and_degree_of_sparse(Y_part,coder,caled_number)

def observe_accuracy(the_D,the_A,the_W):
    coder = SparseCoder(dictionary=the_D.T,transform_alpha=lamda/2., transform_algorithm='lasso_cd')
    indexs=list([])
    for i in range(n_classes):
        label_indexs_part=np.array(np.where(labels==i))[0]
        indexs.extend(label_indexs_part[start_train_number:test_number])
    Y_test=np.array(data[indexs],dtype=float).transpose()/255.
    X_test=(coder.transform(Y_test.T)).T
    H_pred=np.dot(the_W,X_test)
    # Q_pred=np.dot(the_A,X_test)
    print(H_pred)
""" Run 5 times with different random selections """
tab_unlabelled = np.zeros(1)
tab_test = np.zeros(1)

for random_state in range (1):
    # np.random.shuffle(zero_label_indexs)
    index = list([])
    #start
    # for i in range (n_classes):
    #     num_i = 0
    #     j = 0
    #     while num_i < all_number:
    #         j = j + 1
    #         if labels[j] == i:
    #             index.append(j)
    #             num_i = num_i + 1
    #end 10个类别，每个类别有300个下标，数组index总共三千个下标
    """ random selection for training set (20 labelled samples, 80 unlabelled samples) and testing set (100 samples) """
    #start
    for i in range (n_classes):
        index.extend(np.where(labels==i)[0][:all_number])
    index = np.array(index)
    for i in range (n_classes):
        np.random.shuffle(index[all_number*i:all_number*i + all_number])
                
        
    index_l = list([])
    for i in range (n_classes):
        index_l.extend(index[all_number*i:all_number*i +start_train_number])    
    #end 10个类别，每个类别有200个下标，从每个类别的200个下标随机取20个下标，该下标用于labels
    y_labelled = labels[index_l]
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
        index_t.extend(index[all_number*i + start_train_number :all_number*i + start_train_number + test_number])    
    
    y_test = labels[index_t]
    n_test = len(y_test)
    Y_test = np.array(data[index_t],dtype = float).transpose()/255.
    Y_train = np.hstack((Y_labelled,Y_unlabelled))
    """ Preprocessing (make each sample have 5 times norm unity) """
    Y_train = preprocessing.normalize(Y_train.T, norm='l2').T*5
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
    
    n_features = Y_train.shape[0]




    
    """ Start the process, initialize dictionary """
    Ds=list([])
    Bs=np.empty((n_classes,data.shape[1],start_train_number))
    H_Bs=np.empty((n_classes,n_classes,start_train_number))
    Q_Bs=np.empty((n_classes,n_atoms*n_classes,start_train_number))
    Cs=np.empty((n_classes,start_train_number,start_train_number))
    for i in range(n_classes):
        D = initialize_single_D(Y_train, n_atoms, y_labelled,n_labelled,D_index=i)
        D = norm_cols_plus_petit_1(D,c)
        Ds.extend([D])
    D_init=np.copy(Ds[0])
       
    
    print("initializing classifier ... done")
    # caled_number=np.zeros(n_classes,dtype=int)
    # for i in range(n_classes):
    #     caled_number[i]=start_train_number
    for i in range(3000):
        for j in range(n_classes):
            if j==0 and i%100==0:
                print(i)
                sys.stdout.flush()
            # start=(start_train_number+i)*j
            # end=start+(start_train_number+i)
            if j!=0:
                # label_indexs_for_update=np.array(np.where(labels==j))[0][all_number:]
                # np.random.shuffle(label_indexs_for_update)
                # new_index=[label_indexs_for_update[0]]
                # new_y=np.array(data[new_index],dtype = float).transpose()/255.
                # Y_train=np.hstack((Y_train[:,0:end],new_y,Y_train[:,end:]))
                continue
            D=Ds[j]
            coder = SparseCoder(dictionary=D.T,transform_alpha=lamda/2., transform_algorithm='lasso_cd')
            # if j==0 and (i%299==0 or i==0):
                # print("start the observation for dictionary of index "+str(j)+" and i="+str(i))
                # get_part_data_and_observe(coder)
                # D_diff_abs=abs(Ds[0]-D_init)
                # print(D_diff_abs.sum())
            if i==0:
                the_H=np.zeros((n_classes,Y_train.shape[1]),dtype=int) #10,60000
                the_Q=np.zeros((n_atoms*n_classes,Y_train.shape[1]),dtype=int) #2000,60000
                for k in range(Y_train.shape[1]):
                    label=y_labelled[k]
                    the_H[label,k]=1
                    the_Q[n_atoms*label:n_atoms*(label+1),k]=1
                X_single =(coder.transform(Y_train.T[start_train_number*j:start_train_number*j+start_train_number])).T #X_single的每个列向量是一个图像的稀疏表征
                Bs[j]=np.dot(Y_train[:,start_train_number*j:start_train_number*j+start_train_number],X_single.T)
                H_Bs[j]=np.dot(the_H[:,start_train_number*j:start_train_number*j+start_train_number],X_single.T)
                Q_Bs[j]=np.dot(the_Q[:,start_train_number*j:start_train_number*j+start_train_number],X_single.T)
                Cs[j]=np.linalg.inv(np.dot(X_single,X_single.T))
            the_B=Bs[j]
            the_C=Cs[j]
            label_indexs_for_update=np.array(np.where(labels==j))[0][all_number:]
            np.random.shuffle(label_indexs_for_update)
            new_index=[label_indexs_for_update[0]]
            new_y=np.array(data[new_index],dtype = float).transpose()/255.
            new_y=preprocessing.normalize(new_y.T, norm='l2').T*5
            new_y.reshape(n_features,1)
            new_label=labels[new_index]
            new_x=(coder.transform(new_y.T)).T
            new_B=the_B+np.dot(new_y,new_x.T)
            new_C=the_C-(np.matrix(the_C)*np.matrix(new_x)*np.matrix(new_x.T)*np.matrix(the_C))/(np.matrix(new_x.T)*np.matrix(the_C)*np.matrix(new_x)+1) #matrix inversion lemma(Woodbury matrix identity)
            Bs[j]=new_B
            Cs[j]=new_C
            new_D=np.dot(new_B,new_C)
            # new_D = norm_cols_plus_petit_1(new_D,c)
            D=np.copy(new_D)
            Ds[j]=D
            # Y_train=np.hstack((Y_train[:,0:end],new_y,Y_train[:,end:]))

    D_all=np.zeros((data.shape[1],0))
    for i in range(n_classes):
        D_all=np.hstack((D_all,np.copy(Ds[i])))
    with open('D_all_'+str(start_train_number)+'.txt', mode='a+', encoding="utf-8") as w:
        w.seek(0)
        w.truncate()
        w.write(json.dumps(D_all.tolist()))
    print("D_all saved")






    the_H=np.zeros((n_classes,data_count),dtype=int) #10,60000
    the_Q=np.zeros((n_atoms*n_classes,data_count),dtype=int) #2000,60000
    for i in range(data_count):
        label=labels[i]
        the_H[label,i]=1
        the_Q[n_atoms*label:n_atoms*(label+1),i]=1

    D_all = load_local_json_to_obj('D_all_'+str(start_train_number)+'.txt')
    coder = SparseCoder(dictionary=D_all.T,transform_alpha=lamda/2., transform_algorithm='lasso_cd')
    Y_all=np.array(np.copy(data),dtype=float)/255.
    Y_all = preprocessing.normalize(Y_all, norm='l2').T*5
    # X_all=(coder.transform(Y_all.T)).T
    # pdb.set_trace()
    # with open('X_all_'+str(start_train_number)+'.txt', mode='a+', encoding="utf-8") as w:
    #     w.seek(0)
    #     w.truncate()
    #     w.write(json.dumps(X_all.tolist()))
    # print("X_all saved")
    X_all = load_local_json_to_obj('X_all_'+str(start_train_number)+'.txt')
    X_all_T=X_all.T
    X_X_T_inv=np.linalg.inv(np.dot(X_all,X_all_T))
    the_W=np.dot(np.dot(the_H,X_all_T),X_X_T_inv) #10,20000
    the_A=np.dot(np.dot(the_Q,X_all_T),X_X_T_inv) #2000,20000
    pdb.set_trace()
    with open('the_W_'+str(start_train_number)+'.txt', mode='a+', encoding="utf-8") as w:
        w.seek(0)
        w.truncate()
        w.write(json.dumps(the_W.tolist()))
    print("the_W saved")
    with open('the_A_'+str(start_train_number)+'.txt', mode='a+', encoding="utf-8") as w:
        w.seek(0)
        w.truncate()
        w.write(json.dumps(the_A.tolist()))
    print("the_A saved")






    # with open('X_all.txt', mode='a+', encoding="utf-8") as w:
    #     w.write(json.dumps(X_all.tolist()))
    # with open('Y_train.txt', mode='a+', encoding="utf-8") as w:
    #     w.write(json.dumps(Y_train.tolist()))
    # with open('D.txt', mode='a+', encoding="utf-8") as w:
    #     w.write(json.dumps(D.tolist()))
    # with open('W.txt', mode='a+', encoding="utf-8") as w:
    #     w.write(json.dumps(W.tolist()))
    # with open('b.txt', mode='a+', encoding="utf-8") as w:
    #     w.write(json.dumps(b.tolist())) 