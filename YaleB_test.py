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

# def load_img(path):
#     im = Image.open(path)    # 读取文件
#     im_vec=np.asarray(im,dtype=float).T.reshape(-1,1)
#     return im_vec

def write_to_file(path,obj):
    with open(path, mode='a+', encoding="utf-8") as w:
        w.seek(0)
        w.truncate()
        w.write(json.dumps(obj.tolist()))

def cs_omp(y,Phi,N,K):    
    residual=y  #初始化残差
    index=np.zeros(N,dtype=int)
    for i in range(N): #第i列被选中就是1，未选中就是-1
        index[i]= -1
    result=np.zeros((N,1))
    for j in range(K):  #迭代次数
        start_time=time.time()
        product=np.fabs(np.dot(Phi.T,residual))
        pos=np.argmax(product)  #最大投影系数对应的位置        
        index[pos]=1 #对应的位置取1
        my=np.linalg.pinv(Phi[:,index>=0]) #最小二乘          
        a=np.dot(my,y) #最小二乘,看参考文献1     
        residual=y-np.dot(Phi[:,index>=0],a)
        end_time=time.time()
        print(f"part cal var time is: {end_time-start_time} s")
        sys.stdout.flush()
    result[index>=0]=a
    Candidate = np.where(index>=0) #返回所有选中的列
    return  result, Candidate

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

reg_mul=1

t=time.time()

np.random.seed(int(t)%100)

data_count=labels.shape[0]

start_init_number=15
train_number=32
update_times=100
start_test_number=train_number
test_number=32
im_vec_len=w*h

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

maxs_count=100
D_all=np.load('D_all_YaleB_mulD_'+str(w)+'_'+str(h)+'_'+str(update_times)+'.npy')
D_argmaxs=np.empty((maxs_count,D_all.shape[1]),dtype=int)
D_parts=np.empty((maxs_count,D_all.shape[1]))
for i in range(D_all.shape[1]):
    D_argmaxs[:,i]=np.argsort(D_all[:,i])[-maxs_count:]
    D_parts[:,i]=D_all[D_argmaxs[:,i],i]
W_all=np.load('W_all_YaleB_mulD_'+str(w)+'_'+str(h)+'_'+str(update_times)+'.npy')
# D_all=np.load('D_all_YaleB_init'+'.npy')
# W_all=np.load('W_all_YaleB_init'+'.npy')
# A_all=np.load('A_all_YaleB_'+str(update_times))
average_accuracy=0.
for cla in classes:
    indexs=np.array(np.where(labels==cla))[0]
    label_index=lab_to_ind_dir[cla]
    indexs=indexs[start_test_number:start_test_number+test_number]
    Y_test=np.empty((im_vec_len,test_number))
    ind=0
    temp_process=0
    for i in indexs:
        if temp_process%100==0:
            print(temp_process)
            sys.stdout.flush()
        temp_process+=1
        im = Image.open(file_paths[i])    # 读取文件
        im_vec=np.asarray(im,dtype=float).T.reshape(-1,1)
        im_vec=im_vec/255.
        im_vec=im_vec.T[0]
        if im_vec.shape[0]!=im_vec_len:
            print("存在总像素不为307200的图像 程序暂停")
            pdb.set_trace()
        Y_test[:,ind]=im_vec
        ind+=1
    Y_test = preprocessing.normalize(Y_test.T, norm='l2').T*reg_mul
    # Y_test = preprocessing.normalize(Y_test.T, norm='l2').T
    coder = SparseCoder(dictionary=D_all.T,transform_alpha=lamda/2., transform_algorithm='omp')
    # coder = SparseCoder(dictionary=D_all.T,transform_n_nonzero_coefs=30, transform_algorithm='omp')
    X_test=np.empty((D_all.shape[1],Y_test.shape[1]))
    start_time=time.time()
    X_test=(coder.transform(Y_test.T)).T
    end_time=time.time()
    print(f"coder transform time is: {end_time-start_time} s")

    # for i in range(Y_test.shape[1]):
    #     pass
    #     start_time=time.time()
    #     D_used=np.empty(D_all.shape[1])
    #     D_used[:]=-1
    #     # Y_one=Y_test[D_argmaxs[:,i],i]
    #     residual=Y_test[:,i]
    #     X_temp=np.empty(0)
    #     for j in range(210):
    #         start_time_j=time.time()
    #         min_variance=np.inf
    #         min_ind=0
    #         # variances=np.empty(D_all.shape[1])
    #         for k in range(D_all.shape[1]):
    #             pass
    #             if D_used[k]==1:
    #                 continue
    #             ratios=(D_parts[:,k])/residual[D_argmaxs[:,i]]
    #             # ratios=ratios[ratios!=np.nan]
    #             ratios=ratios[ratios!=np.inf]
    #             # ratios=ratios[ratios!=0.]
    #             ratios_mean=ratios.mean()
    #             ratios=ratios/ratios_mean
    #             cur_var=np.var(ratios)
    #             if cur_var<min_variance:
    #                 min_variance=cur_var
    #                 min_ind=k
    #         first_atmo_index=min_ind
    #         D_used[first_atmo_index]=1
    #         D_part=D_all[:,D_used==1]
    #         D_part_pinv=np.linalg.pinv(D_part)
    #         X_temp=np.dot(D_part_pinv,residual)
    #         solved_resi=np.dot(D_part,X_temp)
    #         residual=residual-solved_resi
    #         end_time_j=time.time()
    #         print(f"j is :{j}")
    #         print(f"part cal var time is: {end_time_j-start_time_j} s")
    #         sys.stdout.flush()
    #     X_test[:,i]=0.
    #     X_test[D_used==1,i]=X_temp
    #     end_time=time.time()
    #     print(f"cal var time is: {end_time-start_time} s")


    # for i in range(X_test.shape[1]):
    #     # is_existed_problem=False
    #     # X_sums=list([])
    #     X_one=X_test[:,i]
    #     # temp_part=X_one[label_index*start_init_number:(label_index+1)*start_init_number]
    #     # X_true_sum=abs(temp_part).sum()/temp_part.shape[0]
    #     # X_sums.extend([X_true_sum])
    #     # for j in range(n_classes):
    #     #     if j==label_index:
    #     #         continue
    #     #     other_temp_part=X_one[j*start_init_number:(j+1)*start_init_number]
    #     #     X_other_sum=abs(other_temp_part).sum()/other_temp_part.shape[0]
    #     #     X_sums.extend([X_other_sum])
    #     #     if X_other_sum>=X_true_sum:
    #     #         print(X_other_sum)
    #     #         print(X_true_sum)
    #     #         is_existed_problem=True
    #     # X_sums.sort(reverse=True)
    #     # X_sums=np.array(X_sums)
    #     # if is_existed_problem==True:
    #     #     print(abs(X_one).sum()/420.)
    #     #     print(X_sums)
    #     #     pass
    #     #     pdb.set_trace()
    #     Y_one=Y_test[:,i]
    #     aa=np.dot(D_all.T,Y_one)
    #     bb=aa[label_index*start_init_number:(label_index+1)*start_init_number]
    #     pdb.set_trace()


    the_H=np.dot(W_all,X_test)
    right_num=0.
    for i in range(test_number):
        # pdb.set_trace()
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
    print('label : '+str(cla))
    print('accuracy : '+str(right_num/test_number))
    average_accuracy+=right_num/test_number
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
