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
w=192
h=168
for i in range(40):
    # dir_path="./ExtendedYaleB_"+str(w)+"x"+str(h)+"/"+str(i)
    dir_path="./ExtendedYaleB_"+"300"+"x"+"300"+"_to_"+str(w)+"x"+str(h)+"/"+str(i)
    # dir_path="./ExtendedYaleB_"+"300"+"x"+"300"+"/"+str(i)
    is_not_saw=True
    if os.path.isdir(dir_path):
        n_classes+=1
        classes.extend([i])
        for root, dirs, files in os.walk(dir_path, topdown=False):
            for file_name in files:
                if ".info" in file_name or "Ambient" in file_name:
                    continue
                labels.extend([i])
                file_paths.extend([dir_path+"/"+file_name])
                # if is_not_saw==True:
                #     is_not_saw=False
                #     im = Image.open(dir_path+"/"+file_name)
                #     im.show()
                #     pdb.set_trace()
labels=np.array(labels)
classes=np.array(classes)
file_paths=np.array(file_paths)
for i in range(n_classes):
    lab_to_ind_dir[classes[i]]=i
    ind_to_lab_dir[i]=classes[i]




# ind=0
# for path in file_paths:
#     im = Image.open(path)
#     im = cv2.cvtColor(np.asarray(im),cv2.IMREAD_GRAYSCALE)
#     # im = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
#     label=labels[ind]
#     create_dir_if_not_exist("./ExtendedYaleB_"+"300"+"x"+"300"+"_to_"+str(w)+"x"+str(h))
#     create_dir_if_not_exist("./ExtendedYaleB_"+"300"+"x"+"300"+"_to_"+str(w)+"x"+str(h)+"/"+str(label))
#     im_small=cv2.resize(im,(w,h),interpolation=cv2.INTER_LINEAR)
#     im_small=Image.fromarray(cv2.cvtColor(im_small,cv2.COLOR_RGB2GRAY))
#     im_small.save("./ExtendedYaleB_"+"300"+"x"+"300"+"_to_"+str(w)+"x"+str(h)+"/"+str(label)+path[26:])
#     ind+=1
# pdb.set_trace()





reg_mul=1

t=time.time()

np.random.seed(int(t)%100)

data_count=labels.shape[0]

start_init_number=15
train_number=32
update_times=100
im_vec_len=w*h
transform_n_nonzero_coefs=30

index = list([])
inds_of_file_path_path='inds_of_file_path_wzz_'+str(w)+'_'+str(h)+'_'+str(update_times)+'_'+str(transform_n_nonzero_coefs)+'_'+str(start_init_number)+'.npy'
if os.path.isfile(inds_of_file_path_path):
    inds_of_file_path=np.load(inds_of_file_path_path)
    for i in classes:
        ind_of_lab=lab_to_ind_dir[i]
        labels_of_one_class=inds_of_file_path[ind_of_lab]
        # if i==34 or i==39:    #need to change label rank
        if i==34:    #need to change label rank
            labels_of_one_class.sort()
            # np.random.shuffle(labels_of_one_class)
        if labels_of_one_class.shape[0]<start_init_number:
            print("某个类的样本不足，程序暂停")
            pdb.set_trace()
        inds_of_file_path[ind_of_lab]=labels_of_one_class
else:
    inds_of_file_path=np.empty((n_classes,train_number*2),dtype=int)
    for i in classes:
        ind_of_lab=lab_to_ind_dir[i]
        labels_of_one_class=np.where(labels==i)[0][:train_number*2]
        if labels_of_one_class.shape[0]<start_init_number:
            print("某个类的样本不足，程序暂停")
            pdb.set_trace()
        inds_of_file_path[ind_of_lab]=labels_of_one_class
for i in classes:
    ind_of_lab=lab_to_ind_dir[i]
    index.extend(inds_of_file_path[ind_of_lab][:start_init_number])
index = np.array(index)
# for i in range (n_classes):
#     np.random.shuffle(index[start_init_number*i:start_init_number*i + start_init_number])

index_l = list([])
for i in range (n_classes):
    index_l.extend(index[start_init_number*i:start_init_number*i +start_init_number])
y_labelled = labels[index_l]
n_labelled = len(y_labelled)
Y_labelled=np.zeros((im_vec_len,start_init_number*n_classes))
temp_process=0
ind=0
for i in index_l:
    if temp_process%100==0:
        print(temp_process)
        sys.stdout.flush()
    temp_process+=1
    im_vec=load_img(file_paths[i])
    im_vec=im_vec/255.
    im_vec=im_vec.T[0]
    if im_vec.shape[0]!=im_vec_len:
        print("存在总像素不为307200的图像 程序暂停")
        pdb.set_trace()
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





""" Start the process, initialize dictionary """
Ds=np.empty((n_classes,im_vec_len,n_atoms))
Ws=np.empty((n_classes,n_classes,start_init_number))
As=np.empty((n_classes,n_atoms*n_classes,start_init_number))
# Bs=np.empty((im_vec_len,start_init_number*n_classes))
# H_Bs=np.empty((n_classes,start_init_number*n_classes))
# Q_Bs=np.empty((n_atoms*n_classes,start_init_number*n_classes))
# Cs=np.empty((start_init_number,start_init_number*n_classes))
Cs=None
for i in range(n_classes):
    D = initialize_single_D(Y_init, n_atoms, y_labelled,n_labelled,D_index=i)
    # D = norm_cols_plus_petit_1(D,c)
    Ds[i]=D

print("initializing classifier ... done")
start_t=time.time()

D_all=Ds
D_all=D_all.transpose((0,2,1))
D_all=D_all.reshape(-1,im_vec_len).T
D_all=preprocessing.normalize(D_all.T, norm='l2').T
W_all=Ws
W_all=W_all.transpose((0,2,1))
W_all=W_all.reshape(-1,n_classes).T
A_all=As
A_all=A_all.transpose((0,2,1))
A_all=A_all.reshape(-1,n_classes*n_atoms).T

lambda_init=0.9985
the_lambda=lambda_init
DWA_all=None

for update_index in range(update_times):
    if update_index==0:
        DWA_all,W_all,A_all,Cs=DWA_all_init(D_all,W_all,A_all,n_classes,n_atoms,Y_init,y_labelled,lab_to_ind_dir)
    train(
    DWA_all,D_all,W_all,A_all,Cs,labels,
    file_paths,inds_of_file_path,
    train_number,start_init_number,update_times,update_index,
    n_classes,n_atoms,n_features,lambda_init,the_lambda,transform_n_nonzero_coefs,
    "wzz")
end_t=time.time()
print("train_time : "+str(end_t-start_t))
np.save('D_all_YaleB_wzz_'+str(w)+'_'+str(h)+'_'+str(update_times)+'_'+str(transform_n_nonzero_coefs)+'_'+str(start_init_number),D_all)
print("D_all saved")
np.save('W_all_YaleB_wzz_'+str(w)+'_'+str(h)+'_'+str(update_times)+'_'+str(transform_n_nonzero_coefs)+'_'+str(start_init_number),W_all)
print("W_all saved")
np.save('A_all_YaleB_wzz_'+str(w)+'_'+str(h)+'_'+str(update_times)+'_'+str(transform_n_nonzero_coefs)+'_'+str(start_init_number),A_all)
print("A_all saved")

np.save(inds_of_file_path_path,inds_of_file_path)