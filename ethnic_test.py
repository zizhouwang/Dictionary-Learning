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
import argparse

def load_img(path):
    im = Image.open(path)    # 读取文件
    im_vec=np.asarray(im,dtype=float).T.reshape(-1,1)
    return im_vec

def write_to_file(path,obj):
    with open(path, mode='a+', encoding="utf-8") as w:
        w.seek(0)
        w.truncate()
        w.write(json.dumps(obj.tolist()))

def pre(Y_test,coder,label_index=-1):
    global is_one_test
    global reg_mul
    Y_test=preprocessing.normalize(Y_test.T, norm='l2').T*reg_mul
    X_test=(coder.transform(Y_test.T)).T
    the_H=np.dot(W_all,X_test)
    right_num=0.
    file_paths=list([])
    if is_one_test==True:
        pre=the_H[:,0].argmax()
        dir_path="./"+py_file_name+"_"+str(w)+"x"+str(h)
        dir_path_of_class=dir_path+"/"+str(pre)
        if os.path.isdir(dir_path_of_class):
            for root, dirs, files in os.walk(dir_path_of_class, topdown=False):
                for file_name in files:
                    if ".info" in file_name or "Ambient" in file_name:
                        continue
                    file_paths.extend([dir_path_of_class+"/"+file_name])
        file_paths=np.array(file_paths)
        fea_simis=list([])
        simi_to_im={}
        for path in file_paths:
            im = Image.open(path)    # 读取文件
            im_vec=np.asarray(im,dtype=float).T.reshape(-1,1)
            im_vec=preprocessing.normalize(im_vec.T, norm='l2').T*reg_mul
            im_fea=(coder.transform(im_vec.T)).T
            fea_diff=abs(im_fea-X_test).sum()
            simi_to_im[fea_diff]=im
            fea_simis.extend([fea_diff])
        fea_simis.sort()
        temp_process=0
        for simi in fea_simis:
            simi_im=simi_to_im[simi]
            simi_im.show()
            temp_process+=1
            if temp_process>=5:
                break
        pdb.set_trace()
    else:
        for i in range(Y_test.shape[1]):
            pre=the_H[:,i].argmax()
            if pre==label_index:
                right_num+=1.
            else:
                pass
                # print("start")
                # for j in range(n_classes):
                #     X_one_test=X_test[:,i][j*15:(j+1)*15]
                #     W_one=W_all[:,j*15:(j+1)*15]
                #     print(np.dot(W_one,X_one_test))
                #     pre_one=np.dot(W_one,X_one_test).argmax()
                #     pre_energy=np.dot(W_one,X_one_test)[pre_one]
                #     print(np.dot(W_one,X_one_test)[pre_one])
                #     print(np.dot(W_one,X_one_test).argmax())
                #     print()
                #     pdb.set_trace()
    return right_num

py_file_name="ethnic"

reg_mul=1

t=time.time()

np.random.seed(int(t)%100)

start_init_number=30
train_number=300
update_times=300
start_test_number=start_init_number+train_number
test_number=200

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

n_classes=0
classes=list([])
labels=list([])
file_paths=list([])
lab_to_ind_dir={0:0,1:1,2:2,3:3,4:4,5:5,6:6}
ind_to_lab_dir={0:0,1:1,2:2,3:3,4:4,5:5,6:6}
w=160
h=160
im_vec_len=w*h*3
dir_path="./"+py_file_name+"_"+str(w)+"x"+str(h)+"_test"

D_all=np.load("D_all_"+py_file_name+"_mulD_"+str(w)+"_"+str(h)+"_"+str(update_times)+".npy")
W_all=np.load("W_all_"+py_file_name+"_mulD_"+str(w)+"_"+str(h)+"_"+str(update_times)+".npy")
coder = SparseCoder(dictionary=D_all.T,transform_alpha=lamda/2., transform_algorithm='omp')

is_one_test=True
parser = argparse.ArgumentParser(description='ArgUtils')
parser.add_argument('-pfp', type=str, default='no', help="the file path for pre")
args = parser.parse_args()
if args.pfp=="no":
    is_one_test=False
if is_one_test==True:
    Y_test=load_img(args.pfp)/255.
    pre(Y_test,coder)
    sys.exit()

train_number_of_every_cla=list([])
for i in range(40):
    dir_path_of_class=dir_path+"/"+str(i)
    if os.path.isdir(dir_path_of_class):
        num_of_cla=0
        n_classes+=1
        classes.extend([i])
        for root, dirs, files in os.walk(dir_path_of_class, topdown=False):
            for file_name in files:
                if ".info" in file_name or "Ambient" in file_name:
                    continue
                num_of_cla+=1
                labels.extend([i])
                file_paths.extend([dir_path_of_class+"/"+file_name])
labels=np.array(labels)
classes=np.array(classes)
file_paths=np.array(file_paths)

average_accuracy=0.
true_test_number=test_number
true_start_test_number=start_test_number
for cla in classes:
    indexs=np.array(np.where(labels==cla))[0]
    acl_test_number=indexs.shape[0]
    if acl_test_number<start_test_number+test_number:
        true_test_number=acl_test_number
        true_start_test_number=0
    else:
        true_test_number=test_number
        true_start_test_number=start_test_number
    label_index=lab_to_ind_dir[cla]
    indexs=indexs[true_start_test_number:true_start_test_number+true_test_number]
    if indexs.shape[0]==0:
        continue
    np.random.shuffle(indexs)
    Y_test=np.zeros((im_vec_len,true_test_number))
    pdb.set_trace()
    ind=0
    temp_process=0
    for i in indexs:
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
        Y_test[:,ind]=im_vec
        ind+=1
    # Y_test = preprocessing.normalize(Y_test.T, norm='l2').T
    # coder = SparseCoder(dictionary=D_all.T,transform_n_nonzero_coefs=30, transform_algorithm='omp')
    right_num=pre(Y_test,coder,label_index)


    # X_test=(coder.transform(Y_test.T)).T
    # the_H=np.dot(W_all,X_test)
    # right_num=0.
    # for i in range(true_test_number):
    #     pre=the_H[:,i].argmax()
    #     if pre==label_index:
    #         right_num+=1.
    #     else:
    #         pass
    #         # print("start")
    #         # for j in range(n_classes):
    #         #     X_one_test=X_test[:,i][j*15:(j+1)*15]
    #         #     W_one=W_all[:,j*15:(j+1)*15]
    #         #     print(np.dot(W_one,X_one_test))
    #         #     pre_one=np.dot(W_one,X_one_test).argmax()
    #         #     pre_energy=np.dot(W_one,X_one_test)[pre_one]
    #         #     print(np.dot(W_one,X_one_test)[pre_one])
    #         #     print(np.dot(W_one,X_one_test).argmax())
    #         #     print()
    #         #     pdb.set_trace()




    print('label : '+str(cla))
    print('accuracy : '+str(right_num/true_test_number))
    average_accuracy+=right_num/true_test_number
    sys.stdout.flush()

average_accuracy=average_accuracy/n_classes
print('average_accuracy : '+str(average_accuracy))
# indexs=np.array(np.where(labels!=0))[0]
# np.random.shuffle(indexs)
# indexs=indexs[start_test_number:start_test_number+test_number]
# Y_test=np.zeros((im_vec_len,test_number))
# ind=0
# temp_process=0
# for i in indexs:
#     if temp_process%100==0:
#         print(temp_process)
#         sys.stdout.flush()
#     temp_process+=1
#     im_vec=load_img(file_paths[i])
#     im_vec=im_vec/255.
#     im_vec=im_vec.T[0]
#     if im_vec.shape[0]!=im_vec_len:
#         print("存在总像素不为307200的图像 程序暂停")
#         pdb.set_trace()
#     Y_test[:,ind]=im_vec
#     ind+=1
# Y_test = preprocessing.normalize(Y_test.T, norm='l2').T*5
# coder = SparseCoder(dictionary=D_all.T,transform_alpha=lamda/2., transform_algorithm='lasso_cd')
# X_test=(coder.transform(Y_test.T)).T
# the_H=np.dot(W_all,X_test)
# right_num=0.
# for i in range(test_number):
#     pre=the_H[:,i].argmax()
#     if pre!=0:
#         right_num+=1.
# print('accuracy : '+str(right_num/test_number))
