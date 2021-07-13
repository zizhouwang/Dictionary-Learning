import scipy.io
import time
import pdb
import numpy as np
from SSDL_GU import *
from sklearn.decomposition import SparseCoder
from numpy.linalg import norm
import sys
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from mnist import MNIST
from PIL import Image
import os
import cv2

# data = scipy.io.loadmat('clothes5.mat') # 读取mat文件
data = scipy.io.loadmat('T4.mat') # 读取mat文件
# print(data.keys())  # 查看mat文件中的所有变量
image_vecs=data['test_data']
image_vecs=preprocessing.normalize(image_vecs.T, norm='l2').T
image_vecs=norm_Ys(image_vecs)
labels_mat=data['test_Annotation']
# image_vecs=data['train_data']
# image_vecs=preprocessing.normalize(image_vecs.T, norm='l2').T
# labels_mat=data['train_Annotation']
labels_index=np.empty((labels_mat.shape[0],labels_mat.shape[1]))
labels_index[:]=-1
images_count=np.empty((5),dtype=int)
for i in range(labels_mat.shape[0]):
    one_label_mat=labels_mat[i]
    one_labels_index=np.where(one_label_mat==1)[0]
    labels_index[i,:one_labels_index.shape[0]]=one_labels_index
    images_count[i]=one_labels_index.shape[0]

t=time.time()

np.random.seed(int(t)%100)
n_classes=labels_index.shape[0]
classes=np.arange(n_classes)
# ind_to_lab_dir={0:"仫佬族",1:"纳西族",2:"怒族",3:"普米族",4:"羌族",5:"撒拉族",6:"畲族"}
lab_to_ind_dir={0:0,1:1,2:2,3:3,4:4}
ind_to_lab_dir={0:0,1:1,2:2,3:3,4:4}
w=54
h=46

py_file_name="clothes"

start_init_number=30
train_number=300
im_vec_len=w*h
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
transform_n_nonzero_coefs=30
n_features = image_vecs.shape[0]

# inds_of_file_path_path='inds_of_file_path_wzz_'+py_file_name+'_'+str(w)+'_'+str(h)+'_'+str(update_times)+'_'+str(transform_n_nonzero_coefs)+'_'+str(start_init_number)+'.npy'
# inds_of_file_path=np.load(inds_of_file_path_path)

D_all=np.load("D_all_"+py_file_name+"_mulD_"+str(w)+"_"+str(h)+"_"+".npy")
W_all=np.load("W_all_"+py_file_name+"_mulD_"+str(w)+"_"+str(h)+"_"+".npy")

average_accuracy=0.
# coder = SparseCoder(dictionary=D_all.T,transform_n_nonzero_coefs=transform_n_nonzero_coefs, transform_algorithm="omp")

# for class_index in range(n_classes):
#     Y_test_indexs=labels_index[class_index][:images_count[class_index]]
#     # Y_test_indexs=labels_index[class_index][:train_number]
#     Y_test=image_vecs[:,Y_test_indexs]
#     test_number=Y_test.shape[1]
#     X_test=np.empty((D_all.shape[1],test_number))
#     for i in range(n_classes):
#         D=D_all[:,i*n_atoms:(i+1)*n_atoms]
#         coder = SparseCoder(dictionary=D.T,transform_n_nonzero_coefs=transform_n_nonzero_coefs, transform_algorithm="omp")
#         # pdb.set_trace()
#         X_test_part=(coder.transform(Y_test.T)).T
#         X_test[i*n_atoms:(i+1)*n_atoms]=X_test_part
#     # X_test=(coder.transform(Y_test.T)).T
#     the_H=np.dot(W_all,X_test)
#     right_num=0
#     for i in range(test_number):
#         pre=the_H[:,i].argmax()
#         if class_index==pre:
#             right_num=right_num+1
#     print('label : '+str(class_index))
#     print('accuracy : '+str(right_num*1./test_number))
#     average_accuracy+=right_num/test_number
#     sys.stdout.flush()

# average_accuracy=average_accuracy/n_classes
# print('average_accuracy : '+str(average_accuracy))

Y_test=image_vecs
test_number=Y_test.shape[1]
X_test=np.empty((D_all.shape[1],test_number))
coder = SparseCoder(dictionary=D_all.T,transform_n_nonzero_coefs=transform_n_nonzero_coefs, transform_algorithm="omp")
# X_test=(coder.transform(Y_test.T)).T
X_test=transform(D_all,Y_test,transform_n_nonzero_coefs)
# for i in range(n_classes):
#     D=D_all[:,i*n_atoms:(i+1)*n_atoms]
#     coder = SparseCoder(dictionary=D.T,transform_n_nonzero_coefs=transform_n_nonzero_coefs, transform_algorithm="omp")
#     X_test_part=(coder.transform(Y_test.T)).T
#     X_test[i*n_atoms:(i+1)*n_atoms]=X_test_part
the_H=np.dot(W_all,X_test)
right_num=0
for i in range(test_number):
    pre=the_H[:,i].argmax()
    if labels_mat[pre,i]==1:
        right_num=right_num+1
    else:
        pass
        # print(the_H[:,i])
        # print(labels_mat[:,i])
        # pdb.set_trace()
print('accuracy : '+str(right_num*1./test_number))