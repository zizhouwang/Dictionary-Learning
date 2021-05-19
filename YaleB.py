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
    dir_path="./ExtendedYaleB_"+"300"+"x"+"300"+"_to_"+str(w)+"x"+str(h)+"/"+str(i)
    # dir_path="./ExtendedYaleB/yaleB"+str(i)
    if os.path.isdir(dir_path):
        n_classes+=1
        classes.extend([i])
        for root, dirs, files in os.walk(dir_path, topdown=False):
            for file_name in files:
                if ".info" in file_name or "Ambient" in file_name:
                    continue
                labels.extend([i])
                file_paths.extend([dir_path+"/"+file_name])
labels=np.array(labels)
classes=np.array(classes)
file_paths=np.array(file_paths)
for i in range(n_classes):
    lab_to_ind_dir[classes[i]]=i
    ind_to_lab_dir[i]=classes[i]




# ind=0
# for path in file_paths:
#     im=Image.open(path)
#     label=labels[ind]
#     create_dir_if_not_exist("./ExtendedYaleB_"+str(w)+"x"+str(h))
#     create_dir_if_not_exist("./ExtendedYaleB_"+str(w)+"x"+str(h)+"/"+str(label))
#     x=(640-w)/2
#     y=(480-h)/2
#     im_small=im.crop((x,y,x+w,y+h))
#     im_small.save("./ExtendedYaleB_"+str(w)+"x"+str(h)+"/"+str(label)+path[23:])
#     ind+=1
# pdb.set_trace()





reg_mul=1

t=time.time()

np.random.seed(int(t)%100)

data_count=labels.shape[0]

start_init_number=15
train_number=32
update_times=300
im_vec_len=w*h

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
Y_train = Y_labelled
Y_train = preprocessing.normalize(Y_train.T, norm='l2').T*reg_mul
# Y_train = preprocessing.normalize(Y_train.T, norm='l2').T
n_atoms = start_init_number
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
Ds=np.empty((n_classes,im_vec_len,n_atoms))
Ws=np.empty((n_classes,n_classes,start_init_number))
As=np.empty((n_classes,n_atoms*n_classes,start_init_number))
Bs=np.empty((n_classes,im_vec_len,start_init_number))
H_Bs=np.empty((n_classes,n_classes,start_init_number))
Q_Bs=np.empty((n_classes,n_atoms*n_classes,start_init_number))
Cs=np.empty((n_classes,start_init_number,start_init_number))
for i in range(n_classes):
    D = initialize_single_D(Y_train, n_atoms, y_labelled,n_labelled,D_index=i)
    D = norm_cols_plus_petit_1(D,c)
    Ds[i]=D
D_init=np.copy(Ds[0])
   

print("initializing classifier ... done")
start_t=time.time()
# caled_number=np.zeros(n_classes,dtype=int)
# for i in range(n_classes):
#     caled_number[i]=start_init_number
for i in range(update_times):
    for j in range(n_classes):
        j_label=ind_to_lab_dir[j]
        if j==0 and i%10==0:
            print(i)
            sys.stdout.flush()
        # start=(start_init_number+i)*j
        # end=start+(start_init_number+i)
        D=Ds[j]
        coder = SparseCoder(dictionary=D.T,transform_alpha=lamda/2., transform_algorithm='omp')
        # coder = SparseCoder(dictionary=D.T,transform_n_nonzero_coefs=15, transform_algorithm='omp')
        if i==0:
            the_H=np.zeros((n_classes,Y_train.shape[1]),dtype=int)
            the_Q=np.zeros((n_atoms*n_classes,Y_train.shape[1]),dtype=int)
            for k in range(Y_train.shape[1]):
                label=y_labelled[k]
                lab_index=lab_to_ind_dir[label]
                the_H[lab_index,k]=1
                the_Q[n_atoms*lab_index:n_atoms*(lab_index+1),k]=1
            X_single =(coder.transform(Y_train.T[start_init_number*j:start_init_number*j+start_init_number])).T #X_single的每个列向量是一个图像的稀疏表征
            Bs[j]=np.dot(Y_train[:,start_init_number*j:start_init_number*j+start_init_number],X_single.T)
            H_Bs[j]=np.dot(the_H[:,start_init_number*j:start_init_number*j+start_init_number],X_single.T)#10,200
            Q_Bs[j]=np.dot(the_Q[:,start_init_number*j:start_init_number*j+start_init_number],X_single.T)#2000,200
            Cs[j]=np.linalg.inv(np.dot(X_single,X_single.T))
            Ws[j]=np.dot(H_Bs[j],Cs[j])
            As[j]=np.dot(Q_Bs[j],Cs[j])
        # if j!=0:
        #     continue
        the_B=Bs[j]
        the_H_B=H_Bs[j]
        the_Q_B=Q_Bs[j]
        the_C=Cs[j]
        label_indexs_for_update=np.array(np.where(labels==j_label))[0][:train_number]
        new_index=[label_indexs_for_update[(i+start_init_number)%train_number]]
        im_vec=load_img(file_paths[new_index][0])
        im_vec=im_vec/255.
        new_y=np.array(im_vec,dtype = float)
        new_y=preprocessing.normalize(new_y.T, norm='l2').T*reg_mul
        # new_y=preprocessing.normalize(new_y.T, norm='l2').T
        new_y.reshape(n_features,1)
        new_label=labels[new_index][0]
        new_h=np.zeros((n_classes,1))
        lab_index=lab_to_ind_dir[new_label]
        new_h[lab_index,0]=1
        new_q=np.zeros((n_atoms*n_classes,1))
        new_q[n_atoms*lab_index:n_atoms*(lab_index+1),0]=1
        new_x=(coder.transform(new_y.T)).T
        new_B=the_B+np.dot(new_y,new_x.T)
        new_H_B=the_H_B+np.dot(new_h,new_x.T)
        new_Q_B=the_Q_B+np.dot(new_q,new_x.T)
        new_C=the_C-(np.matrix(the_C)*np.matrix(new_x)*np.matrix(new_x.T)*np.matrix(the_C))/(np.matrix(new_x.T)*np.matrix(the_C)*np.matrix(new_x)+1) #matrix inversion lemma(Woodbury matrix identity)
        Bs[j]=new_B
        H_Bs[j]=new_H_B
        Q_Bs[j]=new_Q_B
        Cs[j]=new_C
        new_D=np.dot(new_B,new_C)
        D=np.copy(new_D)
        Ds[j]=D
        Ws[j]=np.dot(new_H_B,new_C)
        As[j]=np.dot(new_Q_B,new_C)
        # Y_train=np.hstack((Y_train[:,0:end],new_y,Y_train[:,end:]))
end_t=time.time()
print("train_time : "+str(end_t-start_t))
D_all=Ds
D_all=D_all.transpose((0,2,1))
D_all=D_all.reshape(-1,im_vec_len).T
np.save('D_all_YaleB_mulD_'+str(w)+'_'+str(h)+'_'+str(update_times),D_all)
print("D_all saved")
W_all=Ws
W_all=W_all.transpose((0,2,1))
W_all=W_all.reshape(-1,n_classes).T
np.save('W_all_YaleB_mulD_'+str(w)+'_'+str(h)+'_'+str(update_times),W_all)
print("W_all saved")
A_all=As
A_all=A_all.transpose((0,2,1))
A_all=A_all.reshape(-1,n_classes*n_atoms).T
np.save('A_all_YaleB_mulD_'+str(w)+'_'+str(h)+'_'+str(update_times),A_all)
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