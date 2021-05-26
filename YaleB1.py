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


def load_img(path):
    im = Image.open(path)    # 读取文件
    im_vec=np.asarray(im,dtype=float).T.reshape(-1,1)
    return im_vec

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

start_init_number=20
train_number=32
update_times=32
im_vec_len=w*h
transform_n_nonzero_coefs=30

index = list([])
for i in classes:
    labels_of_one_class=np.where(labels==i)[0]
    if labels_of_one_class.shape[0]<start_init_number:
        print("某个类的样本不足，程序暂停")
        pdb.set_trace()
    index.extend(np.where(labels==i)[0][:start_init_number])
index = np.array(index)
for i in range (n_classes):
    np.random.shuffle(index[start_init_number*i:start_init_number*i + start_init_number])

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
Bs=np.empty((im_vec_len,start_init_number*n_classes))
H_Bs=np.empty((n_classes,start_init_number*n_classes))
Q_Bs=np.empty((n_atoms*n_classes,start_init_number*n_classes))
Cs=np.empty((start_init_number,start_init_number*n_classes))
for i in range(n_classes):
    D = initialize_single_D(Y_init, n_atoms, y_labelled,n_labelled,D_index=i)
    # D = norm_cols_plus_petit_1(D,c)
    Ds[i]=D
D_init=np.copy(Ds[0])
   

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

# caled_number=np.zeros(n_classes,dtype=int)
# for i in range(n_classes):
#     caled_number[i]=start_init_number
lambda_init=0.9985
the_lambda=lambda_init
DWA_all=None
for i in range(update_times):
    if i==0:
        coder = SparseCoder(dictionary=D_all.T,transform_n_nonzero_coefs=transform_n_nonzero_coefs, transform_algorithm='omp')
        the_H=np.zeros((n_classes,Y_init.shape[1]),dtype=int)
        the_Q=np.zeros((n_atoms*n_classes,Y_init.shape[1]),dtype=int)
        for k in range(Y_init.shape[1]):
            label=y_labelled[k]
            lab_index=lab_to_ind_dir[label]
            the_H[lab_index,k]=1
            the_Q[n_atoms*lab_index:n_atoms*(lab_index+1),k]=1
        X_single =(coder.transform(Y_init.T)).T #X_single的每个列向量是一个图像的稀疏表征
        Bs=np.dot(Y_init,X_single.T)
        H_Bs=np.dot(the_H,X_single.T)
        Q_Bs=np.dot(the_Q,X_single.T)
        Cs=np.linalg.inv(np.dot(X_single,X_single.T))
        W_all=np.dot(H_Bs,Cs)
        A_all=np.dot(Q_Bs,Cs)
        DWA_all=np.vstack((D_all,W_all,A_all))
    for j in range(n_classes):
        j_label=ind_to_lab_dir[j]
        # if j==0 and i%10==0:
        #     print(i)
        #     sys.stdout.flush()
        coder = SparseCoder(dictionary=D_all.T,transform_n_nonzero_coefs=transform_n_nonzero_coefs, transform_algorithm='omp')
        label_indexs_for_update=np.array(np.where(labels==j_label))[0][:train_number]
        new_index=[label_indexs_for_update[(i+start_init_number)%32]]
        new_label=labels[new_index][0]
        lab_index=lab_to_ind_dir[new_label]
        im_vec=load_img(file_paths[new_index][0])
        print(file_paths[new_index][0])
        sys.stdout.flush()
        im_vec=im_vec/255.
        new_y=np.array(im_vec,dtype = float)
        new_y=preprocessing.normalize(new_y.T, norm='l2').T*reg_mul
        new_y.reshape(n_features,1)
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

        # the_B=Bs
        # the_H_B=H_Bs
        # the_Q_B=Q_Bs
        # new_B=the_B+np.dot(new_y,new_x.T)
        # new_H_B=the_H_B+np.dot(new_h,new_x.T)
        # new_Q_B=the_Q_B+np.dot(new_q,new_x.T)
        # new_C=the_C-(np.matrix(the_C)*np.matrix(new_x)*np.matrix(new_x.T)*np.matrix(the_C))/(np.matrix(new_x.T)*np.matrix(the_C)*np.matrix(new_x)+1)
        # Bs=new_B
        # H_Bs=new_H_B
        # Q_Bs=new_Q_B
        # Cs=new_C
        # new_D=np.dot(new_B,new_C)
        # D_all=new_D
        # W_all=np.dot(new_H_B,new_C)
        # A_all=np.dot(new_Q_B,new_C)
    part_lambda=(1-i/update_times)
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
# D_all=Ds
# D_all=D_all.transpose((0,2,1))
# D_all=D_all.reshape(-1,im_vec_len).T
np.save('D_all_YaleB_true_'+str(w)+'_'+str(h)+'_'+str(update_times)+'_'+str(transform_n_nonzero_coefs)+'_'+str(start_init_number),D_all)
print("D_all saved")
# W_all=Ws
# W_all=W_all.transpose((0,2,1))
# W_all=W_all.reshape(-1,n_classes).T
np.save('W_all_YaleB_true_'+str(w)+'_'+str(h)+'_'+str(update_times)+'_'+str(transform_n_nonzero_coefs)+'_'+str(start_init_number),W_all)
print("W_all saved")
# A_all=As
# A_all=A_all.transpose((0,2,1))
# A_all=A_all.reshape(-1,n_classes*n_atoms).T
np.save('A_all_YaleB_true_'+str(w)+'_'+str(h)+'_'+str(update_times)+'_'+str(transform_n_nonzero_coefs)+'_'+str(start_init_number),A_all)
print("A_all saved")

    # D_all=np.zeros((data.shape[1],0))
    # for i in range(n_classes):
    #     D_all=np.hstack((D_all,np.copy(Ds[i])))
    # np.save('D_all_YaleB_'+str(update_times),D_all)
    # print("D_all saved")
    # W_all=np.zeros((Ws.shape[1],0))
    # for i in range(n_classes):
    #     W_all=np.hstack((W_all,np.copy(Ws[i])))
    # np.save('W_all_YaleB_'+str(update_times),W_all)
    # print("W_all saved")
    # A_all=np.zeros((As.shape[1],0))
    # for i in range(n_classes):
    #     A_all=np.hstack((A_all,np.copy(As[i])))
    # np.save('A_all_YaleB_'+str(update_times),A_all)
    # print("A_all saved")