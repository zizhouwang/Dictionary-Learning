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

def load_img(path):
    im = Image.open(path)    # 读取文件
    im_vec=np.asarray(im,dtype=float).T.reshape(-1,1)
    return im_vec

def write_to_file(path,obj):
    with open(path, mode='a+', encoding="utf-8") as w:
        w.seek(0)
        w.truncate()
        w.write(json.dumps(obj.tolist()))

def transform_var(Y_one,D_all,D_argmaxs):
    global n_atoms
    global n_classes
    global transform_n_nonzero_coefs
    X_one_test=np.empty((n_atoms*n_classes,1))
    start_time=time.time()
    D_used=np.empty(D_all.shape[1])
    D_used[:]=-1
    residual=Y_one
    X_temp=np.empty(0)
    for j in range(30):
        start_time_j=time.time()
        min_variance=np.inf
        min_ind=0
        # variances=np.empty(D_all.shape[1])
        for k in range(D_all.shape[1]):
            pass
            if D_used[k]==1:
                continue
            ratios=(D_parts[:,k])/residual[D_argmaxs[:,k]]
            # ratios=ratios[ratios!=np.nan]
            ratios=ratios[ratios!=np.inf]
            # ratios=ratios[ratios!=0.]
            ratios_mean=ratios.mean()
            ratios=ratios/ratios_mean
            cur_var=np.var(ratios)
            if cur_var<min_variance:
                min_variance=cur_var
                min_ind=k
        first_atmo_index=min_ind
        D_used[first_atmo_index]=1
        D_part=D_all[:,D_used==1]
        D_part_pinv=np.linalg.pinv(D_part)
        X_temp=np.dot(D_part_pinv,residual)
        solved_resi=np.dot(D_part,X_temp)
        residual=residual-solved_resi
        end_time_j=time.time()
        sys.stdout.flush()
    X_one_test[:,0]=0.
    X_one_test[D_used==1,0]=X_temp
    end_time=time.time()
    print(abs(residual).sum())
    return X_one_test

def transform_normal(Y_one,D_all,D_argmaxs):
    global n_atoms
    global n_classes
    global transform_n_nonzero_coefs
    X_one_test=np.empty((n_atoms*n_classes,1))
    start_time=time.time()
    D_used=np.empty(D_all.shape[1])
    D_used[:]=-1
    residual=Y_one
    X_temp=np.empty(0)
    for j in range(30):
        start_time_j=time.time()
        min_variance=np.inf
        min_ind=0
        aa=np.dot(residual,D_all).argsort()
        for k in range(D_all.shape[1]):
            min_ind=aa[D_all.shape[1]-k-1]
            break
        first_atmo_index=min_ind
        D_used[first_atmo_index]=1
        D_part=D_all[:,D_used==1]
        D_part_pinv=np.linalg.pinv(D_part)
        X_temp=np.dot(D_part_pinv,residual)
        solved_resi=np.dot(D_part,X_temp)
        residual=residual-solved_resi
        end_time_j=time.time()
        sys.stdout.flush()
    X_one_test[:,0]=0.
    X_one_test[D_used==1,0]=X_temp
    end_time=time.time()
    print(abs(residual).sum())
    return X_one_test

n_classes=0
classes=list([])
labels=list([])
file_paths=list([])
lab_to_ind_dir={}
ind_to_lab_dir={}
w=1024
h=1024
dir_path="./font"+"/"
is_not_saw=True
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

reg_mul=1

t=time.time()

np.random.seed(int(t)%100)

data_count=labels.shape[0]

start_init_number=30
train_number=32
start_test_number=train_number
test_number=50
im_vec_len=w*h
transform_n_nonzero_coefs=30

""" Parameters in optimization  """
n_atoms = start_init_number
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

maxs_count=100
D_all=np.load('model/D_all_textt_recog_wzz_'+str(w)+'_'+str(h)+'_'+str(transform_n_nonzero_coefs)+'_'+str(start_init_number)+'.npy')
D_argmaxs=np.empty((maxs_count,D_all.shape[1]),dtype=int)
D_parts=np.empty((maxs_count,D_all.shape[1]))
for i in range(D_all.shape[1]):
    D_argmaxs[:,i]=np.argsort(D_all[:,i])[-maxs_count:]
    D_parts[:,i]=D_all[D_argmaxs[:,i],i]
W_all=np.load('model/W_all_textt_recog_wzz_'+str(w)+'_'+str(h)+'_'+str(transform_n_nonzero_coefs)+'_'+str(start_init_number)+'.npy')
average_accuracy=0.
indexs=np.arange(test_number)
Y_test=np.empty((im_vec_len,test_number))
ind=0
temp_process=0
for i in indexs:
    temp_process+=1
    im_vec=load_img_black_white(file_paths[i])
    im_vec=im_vec/255.
    im_vec=im_vec.T[0]
    Y_test[:,ind]=im_vec
    ind+=1
Y_test = preprocessing.normalize(Y_test.T, norm='l2').T*reg_mul
Y_test=norm_Ys(Y_test)

coder = SparseCoder(dictionary=D_all.T,transform_n_nonzero_coefs=transform_n_nonzero_coefs, transform_algorithm='omp')

X_test = (coder.transform(Y_test.T)).T
the_H=np.dot(W_all,X_test)
right_num=0.
for i in range(test_number):
    label=ind_to_lab_dir[i]
    pre=the_H[:,i].argmax()
    if pre==i:
        right_num+=1.
    else:
        pass
average_accuracy=right_num/test_number
print('accuracy : '+str(average_accuracy))
