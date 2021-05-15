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
import hashlib
import cv2
import numpy as np

def add_noise(im):
    peppers=cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR)
    row, column, deep = peppers.shape
    noise_salt = np.random.randint(0, 256, (row, column))
    flag = 0.2#噪点比例
    noise_salt = np.where(noise_salt < flag * 256, 255, 0)
    noise_salt=np.stack((noise_salt,noise_salt,noise_salt),axis=2)
    peppers.astype("float")
    noise_salt.astype("float")
    salt = peppers + noise_salt
    salt = np.where(salt > 255, 255, salt)
    salt=salt.astype("uint8")
    image = Image.fromarray(cv2.cvtColor(salt,cv2.COLOR_BGR2RGB)) 
    return image

def get_image_md5(path):
    file = open(path, "rb")
    md = hashlib.md5()
    md.update(file.read())
    return md.hexdigest()

def create_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)


def load_img(path):
    im = Image.open(path)    # 读取文件
    im_vec=np.asarray(im,dtype=float).T.reshape(-1,1)
    return im_vec

def write_to_file(path,obj):
    with open(path, mode="a+", encoding="utf-8") as w:
        w.seek(0)
        w.truncate()
        w.write(json.dumps(obj.tolist()))

def get_test_data(file_paths):
    ind=0
    for path in file_paths:
        im=Image.open(path)
        im = cv2.cvtColor(np.asarray(im),cv2.IMREAD_COLOR)
        im=im*0.5
        im=im.astype(np.uint8)
        im_small=cv2.resize(im,(w,h),interpolation=cv2.INTER_LINEAR)
        im_small=add_noise(im_small)
        label=labels[ind]
        create_dir_if_not_exist("./"+py_file_name+"_"+str(w)+"x"+str(h)+"_test")
        create_dir_if_not_exist("./"+py_file_name+"_"+str(w)+"x"+str(h)+"_test"+"/"+str(label))
        im_md5=get_image_md5(path)
        # im=Image.fromarray(im)
        im_small.save("./"+py_file_name+"_"+str(w)+"x"+str(h)+"_test"+"/"+str(label)+"/"+im_md5+".png")
        ind+=1
    pdb.set_trace()

py_file_name="ethnic"

n_classes=0
classes=list([])
labels=list([])
file_paths=list([])
# ind_to_lab_dir={0:"仫佬族",1:"纳西族",2:"怒族",3:"普米族",4:"羌族",5:"撒拉族",6:"畲族"}
lab_to_ind_dir={0:0,1:1,2:2,3:3,4:4,5:5,6:6}
ind_to_lab_dir={0:0,1:1,2:2,3:3,4:4,5:5,6:6}
w=160
h=160
train_number_of_every_cla=list([])
for i in range(8):
    dir_path="./"+py_file_name+"_"+str(w)+"x"+str(h)+"/"+str(i)
    # dir_path="./"+py_file_name+"_original/"+str(i)
    if os.path.isdir(dir_path):
        num_of_cla=0
        n_classes+=1
        classes.extend([i])
        for root, dirs, files in os.walk(dir_path, topdown=False):
            for file_name in files:
                if ".info" in file_name or "Ambient" in file_name:
                    continue
                num_of_cla+=1
                labels.extend([i])
                file_paths.extend([dir_path+"/"+file_name])
            train_number_of_every_cla.extend([num_of_cla])
labels=np.array(labels)
classes=np.array(classes)
file_paths=np.array(file_paths)
train_number_of_every_cla=np.array(train_number_of_every_cla)



# get_test_data(file_paths)
# ind=0
# for path in file_paths:
#     im=Image.open(path)
#     im = cv2.cvtColor(np.asarray(im),cv2.IMREAD_COLOR)
#     label=labels[ind]
#     create_dir_if_not_exist("./"+py_file_name+"_"+str(w)+"x"+str(h))
#     create_dir_if_not_exist("./"+py_file_name+"_"+str(w)+"x"+str(h)+"/"+str(label))
#     im_md5=get_image_md5(path)
#     im_small=cv2.resize(im,(w,h),interpolation=cv2.INTER_LINEAR)
#     im_small=Image.fromarray(im_small)
#     # pdb.set_trace()
#     im_small.save("./"+py_file_name+"_"+str(w)+"x"+str(h)+"/"+str(label)+"/"+im_md5+".png")
#     ind+=1
# pdb.set_trace()





reg_mul=1

t=time.time()

np.random.seed(int(t)%100)

data_count=labels.shape[0]

start_init_number=30
train_number=300
update_times=300
im_vec_len=w*h*3


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
Y_train = preprocessing.normalize(Y_train.T, norm="l2").T*reg_mul
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
        # coder = SparseCoder(dictionary=D.T,transform_alpha=lamda/2., transform_algorithm="omp")
        coder = SparseCoder(dictionary=D.T,transform_n_nonzero_coefs=30, transform_algorithm="omp")
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
            H_Bs[j]=np.dot(the_H[:,start_init_number*j:start_init_number*j+start_init_number],X_single.T)
            Q_Bs[j]=np.dot(the_Q[:,start_init_number*j:start_init_number*j+start_init_number],X_single.T)
            Cs[j]=np.linalg.inv(np.dot(X_single,X_single.T))
            Ws[j]=np.dot(H_Bs[j],Cs[j])
            As[j]=np.dot(Q_Bs[j],Cs[j])

            # the_H=np.zeros((n_classes,Y_train.shape[1]),dtype=int)
            # the_H[:]=0.0
            # the_Q=np.zeros((n_atoms*n_classes,Y_train.shape[1]),dtype=int)
            # the_Q[:]=0.0
            # data_indexs=np.array(np.where(y_labelled==j_label))[0]
            # the_H[j,data_indexs]=1
            # the_Q[n_atoms*j:n_atoms*(j+1),data_indexs]=1
            # X_single =(coder.transform(Y_train.T)).T #X_single的每个列向量是一个图像的稀疏表征
            # Bs[j]=np.dot(Y_train[:,start_init_number*j:start_init_number*j+start_init_number],X_single.T[start_init_number*j:start_init_number*j+start_init_number])
            # H_Bs[j]=np.dot(the_H,X_single.T)
            # Q_Bs[j]=np.dot(the_Q,X_single.T)
            # Cs[j]=np.linalg.inv(np.dot(X_single,X_single.T))
            # Ws[j]=np.dot(H_Bs[j],Cs[j])
            # As[j]=np.dot(Q_Bs[j],Cs[j])
        the_B=Bs[j]
        the_H_B=H_Bs[j]
        the_Q_B=Q_Bs[j]
        the_C=Cs[j]
        label_indexs_for_update=np.array(np.where(labels==j_label))[0][:train_number]
        num_of_cla=train_number_of_every_cla[j]
        if num_of_cla>start_init_number+train_number:
            num_of_cla=start_init_number+train_number
        new_index=[label_indexs_for_update[(i+start_init_number)%num_of_cla]]
        im_vec=load_img(file_paths[new_index][0])
        im_vec=im_vec/255.
        new_y=np.array(im_vec,dtype = float)
        new_y=preprocessing.normalize(new_y.T, norm="l2").T*reg_mul
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
np.save("D_all_"+py_file_name+"_mulD_"+str(w)+"_"+str(h)+"_"+str(update_times),D_all)
print("D_all saved")
W_all=Ws
W_all=W_all.transpose((0,2,1))
W_all=W_all.reshape(-1,n_classes).T
np.save("W_all_"+py_file_name+"_mulD_"+str(w)+"_"+str(h)+"_"+str(update_times),W_all)
print("W_all saved")
A_all=As
A_all=A_all.transpose((0,2,1))
A_all=A_all.reshape(-1,n_classes*n_atoms).T
np.save("A_all_"+py_file_name+"_mulD_"+str(w)+"_"+str(h)+"_"+str(update_times),A_all)
print("A_all saved")

    # D_all=np.zeros((data.shape[1],0))
    # for i in range(n_classes):
    #     D_all=np.hstack((D_all,np.copy(Ds[i])))
    # np.save("D_all_YaleB_"+str(update_times),D_all)
    # print("D_all saved")
    # W_all=np.zeros((Ws.shape[1],0))
    # for i in range(n_classes):
    #     W_all=np.hstack((W_all,np.copy(Ws[i])))
    # np.save("W_all_YaleB_"+str(update_times),W_all)
    # print("W_all saved")
    # A_all=np.zeros((As.shape[1],0))
    # for i in range(n_classes):
    #     A_all=np.hstack((A_all,np.copy(As[i])))
    # np.save("A_all_YaleB_"+str(update_times),A_all)
    # print("A_all saved")