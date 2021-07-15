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
image_vecs=data['train_data']
image_vecs=preprocessing.normalize(image_vecs.T, norm='l2').T
image_vecs=norm_Ys(image_vecs)
labels_mat=data['train_Annotation']
labels_mat=labels_mat*2-1
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

py_file_name="src_clothes"

start_init_number=30
train_number=300
update_times=400
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

inds_of_file_path_path='inds_of_file_path_wzz_'+py_file_name+'_'+str(w)+'_'+str(h)+'_'+str(transform_n_nonzero_coefs)+'_'+str(start_init_number)+'.npy'
if os.path.isfile(inds_of_file_path_path):
    inds_of_file_path=np.load(inds_of_file_path_path)
    for class_index in classes:
        labels_of_one_class=inds_of_file_path[class_index][:images_count[class_index]]
        # if i==34 or i==39:    #need to change label rank
        if class_index==34:    #need to change label rank
            labels_of_one_class.sort()
            # np.random.shuffle(labels_of_one_class)
        if labels_of_one_class.shape[0]<start_init_number:
            print("某个类的样本不足，程序暂停")
            pdb.set_trace()
        inds_of_file_path[class_index][:images_count[class_index]]=labels_of_one_class
else:
    inds_of_file_path=np.empty((n_classes,labels_index.shape[1]),dtype=int)
    for class_index in classes:
        labels_of_one_class=labels_index[class_index][:images_count[class_index]]
        if labels_of_one_class.shape[0]<start_init_number:
            print("某个类的样本不足，程序暂停")
            pdb.set_trace()
        inds_of_file_path[class_index][:images_count[class_index]]=labels_of_one_class

""" Start the process, initialize dictionary """
Ds=np.empty((n_classes,im_vec_len,n_atoms))
Ws=np.empty((n_classes,n_classes,start_init_number))
As=np.empty((n_classes,n_atoms*n_classes,start_init_number))
Bs=np.empty((n_classes,im_vec_len,start_init_number))
H_Bs=np.empty((n_classes,n_classes,start_init_number))
Q_Bs=np.empty((n_classes,n_atoms*n_classes,start_init_number))
Cs=np.empty((n_classes,start_init_number,start_init_number))
for class_index in range(n_classes):
    D = image_vecs[:,inds_of_file_path[class_index][:start_init_number]]
    D = norm_cols_plus_petit_1(D,c)
    Ds[class_index]=np.copy(D)

print("initializing classifier ... done")
start_t=time.time()
end_t=time.time()
print("train_time : "+str(end_t-start_t))
D_all=Ds
D_all=D_all.transpose((0,2,1))
D_all=D_all.reshape(-1,im_vec_len).T
D_all=preprocessing.normalize(D_all.T, norm='l2').T
np.save("D_all_"+py_file_name+"_mulD_"+str(w)+"_"+str(h)+"_",D_all)
print("D_all saved")
W_all=Ws
W_all=W_all.transpose((0,2,1))
W_all=W_all.reshape(-1,n_classes).T
np.save("W_all_"+py_file_name+"_mulD_"+str(w)+"_"+str(h)+"_",W_all)
print("W_all saved")
A_all=As
A_all=A_all.transpose((0,2,1))
A_all=A_all.reshape(-1,n_classes*n_atoms).T
np.save("A_all_"+py_file_name+"_mulD_"+str(w)+"_"+str(h)+"_",A_all)
print("A_all saved")

np.save(inds_of_file_path_path,inds_of_file_path)