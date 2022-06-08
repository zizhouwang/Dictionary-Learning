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
from PIL import Image
import os
import cv2
from sift import *
from cnn_mnist_pytorch import *

is_sift=True

def load_feature(file_path,length=None):
    if is_sift:
        res=sift_fea([file_path],2)[0]
    else:
        res=get_feature(file_path)
    if length is not None:
        if res.shape[0]>length:
            res=res[:length]
        else:
            temp=np.random.rand(length)
            temp[:res.shape[0]]=res
            res=temp
    return np.array([res]).T

def create_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)

def write_to_file(path,obj):
    with open(path, mode='a+', encoding="utf-8") as w:
        w.seek(0)
        w.truncate()
        w.write(json.dumps(obj.tolist()))

n_classes=0
classes=list([])
labels=list([])
file_paths=list([])
lab_to_ind_dir={}
ind_to_lab_dir={}
dir_path="./font"
if os.path.isdir(dir_path):
    for root, dirs, files in os.walk(dir_path, topdown=False):
        for file_name in files:
            if ".info" in file_name or "Ambient" in file_name:
                continue
            classes.extend([n_classes])
            labels.extend([n_classes])
            ind_to_lab_dir[n_classes]=file_name[0]
            lab_to_ind_dir[file_name[0]]=n_classes
            file_paths.extend([dir_path+"/"+file_name])
            n_classes+=1
labels=np.array(labels)
classes=np.array(classes)
file_paths=np.array(file_paths)
example_data=load_feature(file_paths[0])
im_vec_len=example_data.shape[0]
reg_mul=1

t=time.time()

np.random.seed(int(t)%100)

data_count=labels.shape[0]

start_init_number=60
train_number=32
update_times=10
# w=1024
# h=1024
# im_vec_len=w*h
transform_n_nonzero_coefs=30

Y_labelled=np.empty((im_vec_len,start_init_number))
temp_process=0
ind=0
for i in range(start_init_number):
    if temp_process%100==0:
        print(temp_process)
        sys.stdout.flush()
    temp_process+=1
    im_vec=load_feature(file_paths[-1-i],im_vec_len)
    im_vec=im_vec.T[0]
    Y_labelled[:,ind]=im_vec
    ind+=1
Y_init = Y_labelled
Y_init = preprocessing.normalize(Y_init.T, norm='l2').T*reg_mul
Y_init=norm_Ys(Y_init)
n_atoms = start_init_number
n_neighbor = 8
lamda = 0.5
beta = 1.
r = 2.
c = 1.

seed = 0 # to save the way initialize dictionary
n_iter_sp = 50 #number max of iteration in sparse coding
n_iter_du = 50 # number max of iteration in dictionary update
n_iter = 15 # number max of general iteration

n_features = Y_init.shape[0]

Cs=None

print("initializing classifier ... done")
start_t=time.time()

D_all=np.copy(Y_init[:,n_atoms*0:n_atoms*0+n_atoms])
D_all=preprocessing.normalize(D_all.T, norm='l2').T

lambda_init=0.9985
the_lambda=lambda_init
DWA_all=None

for update_index in range(update_times):
    print('update_index:'+str(update_index))
    if update_index==0:
        the_H = np.zeros((n_classes, Y_init.shape[1]), dtype=int)
        the_Q = np.zeros((n_atoms * n_classes, Y_init.shape[1]), dtype=int)
        for k in range(Y_init.shape[1]):
            label = ind_to_lab_dir[n_classes-k-1]
            lab_index = lab_to_ind_dir[label]
            the_H[lab_index, k] = 1
            the_Q[n_atoms * lab_index:n_atoms * (lab_index + 1), k] = 1
        X_single = np.zeros((D_all.shape[1], D_all.shape[1]), dtype=float)
        for j in range(D_all.shape[1]):
            X_single[j][j] = 1.
        H_Bs = np.dot(the_H, X_single.T)
        Q_Bs = np.dot(the_Q, X_single.T)
        Cs = np.linalg.inv(np.dot(X_single, X_single.T))
        W_all = np.dot(H_Bs, Cs)
        A_all = np.dot(Q_Bs, Cs)
        DWA_all = np.vstack((D_all, W_all, A_all))
    for j in range(n_classes):
        if j%10==0:
            print(j)
            sys.stdout.flush()
        # sift 原子 标签 训练时间  准确率 迭代次数
        #      30  50     272秒  0.98  5
        #      30  200    1136秒 0.525 5
        #      30  200    2073秒 0.49 10
        #      30  1000   4967秒 0.124 5
        #      60  200    7499秒 0.98 10
        # cnn  原子 标签 训练时间  准确率 迭代次数
        #      30  200    2176秒  0.295  10
        if j>200:
            break
        coder = SparseCoder(dictionary=D_all.T,transform_n_nonzero_coefs=transform_n_nonzero_coefs, transform_algorithm='omp')
        label=ind_to_lab_dir[j]
        lab_index=j
        if lab_index>=n_classes:
            break
        im_vec=load_feature(file_paths[j],im_vec_len)
        # im_vec=load_img_black_white(file_paths[j])
        new_y=np.array(im_vec,dtype = float)
        new_y=preprocessing.normalize(new_y.T, norm='l2').T
        new_y=norm_Ys(new_y)
        new_y=new_y.reshape(n_features,1)
        new_h=np.zeros((n_classes,1))
        new_h[lab_index,0]=1
        new_q=np.zeros((n_atoms*n_classes,1))
        new_q[n_atoms*lab_index:n_atoms*(lab_index+1),0]=1
        new_yhq=np.vstack((new_y,new_h,new_q))
        new_x=(coder.transform(new_y.T)).T
        the_C=Cs
        the_u=(1/the_lambda)*np.dot(the_C,new_x)
        gamma=1/(1+np.dot(new_x.T,the_u))
        the_r=new_yhq-np.dot(DWA_all,new_x)
        new_C=(1/the_lambda)*the_C-gamma*np.dot(the_u,the_u.T)
        new_DWA=DWA_all+gamma*np.dot(the_r,the_u.T)
        DWA_all=new_DWA
    part_lambda=(1-update_index/update_times)
    the_lambda=1-(1-lambda_init)*part_lambda*part_lambda*part_lambda
    D_all=DWA_all[0:D_all.shape[0],:]
    W_all=DWA_all[D_all.shape[0]:D_all.shape[0]+W_all.shape[0],:]
    A_all=DWA_all[D_all.shape[0]+W_all.shape[0]:,:]
    D_all=preprocessing.normalize(D_all.T, norm='l2').T
    W_all=preprocessing.normalize(W_all.T, norm='l2').T
    A_all=preprocessing.normalize(A_all.T, norm='l2').T
    DWA_all=np.vstack((D_all,W_all,A_all))
end_t=time.time()
print("train_time : "+str(end_t-start_t))
np.save('model/D_all_textt_recog_wzz_'+str(transform_n_nonzero_coefs)+'_'+str(start_init_number),D_all)
print("D_all saved")
np.save('model/W_all_textt_recog_wzz_'+str(transform_n_nonzero_coefs)+'_'+str(start_init_number),W_all)
print("W_all saved")
np.save('model/A_all_textt_recog_wzz_'+str(transform_n_nonzero_coefs)+'_'+str(start_init_number),A_all)
print("A_all saved")

# np.save(inds_of_file_path_path,inds_of_file_path)