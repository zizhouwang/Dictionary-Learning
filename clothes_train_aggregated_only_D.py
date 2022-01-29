import scipy.io
import time
import pdb
import numpy as np
from SSDL_GU import *
from MIQP import *
from sklearn.decomposition import SparseCoder
from numpy.linalg import norm
from numpy import linalg as LA
import sys
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from mnist import MNIST
from PIL import Image
import math
import os
import cv2
from learning_incoherent_dictionary import *
from numpy import random

data = scipy.io.loadmat('T4.mat') # 读取mat文件
image_vecs=data['train_data']
labels_mat=data['train_Annotation']
labels_mat=labels_mat*2-1
change_num=1e-7
image_vecs=preprocessing.normalize(image_vecs.T, norm='l2').T
image_vecs=norm_Ys(image_vecs)
images_count=np.empty((5),dtype=int)
for i in range(labels_mat.shape[0]):
    one_label_mat=labels_mat[i]
    one_labels_index=np.where(one_label_mat==1)[0]
    images_count[i]=one_labels_index.shape[0]
t=time.time()

np.random.seed(int(t)%100)
n_classes=labels_mat.shape[0]
classes=np.arange(n_classes)
w=54
h=46

py_file_name="clothes"

start_init_number=30
train_number=300
update_times=10
im_vec_len=w*h
n_atoms = 300
transform_n_nonzero_coefs=45
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
n_features = image_vecs.shape[0]

""" Start the process, initialize dictionary """
Ds=np.empty((n_classes,im_vec_len,n_atoms))
As=np.empty((n_atoms*n_classes,n_atoms))
Bs=np.empty((im_vec_len,n_atoms))
Cs=np.empty((start_init_number,n_atoms))
D=np.empty((im_vec_len,n_atoms))
# for class_index in range(n_classes):
#     D[:,:start_init_number] = image_vecs[:,inds_of_file_path[class_index][:start_init_number]]
#     # D=random.random(size=(D.shape[0],D.shape[1]))
#     D = norm_cols_plus_petit_1(D,c)
#     Ds[class_index]=np.copy(D)
# D=Ds.transpose((0,2,1)).reshape(-1,im_vec_len).T
D=np.random.rand(im_vec_len,n_atoms)
D = preprocessing.normalize(D.T, norm='l2').T
Ds=D
print("initializing classifier ... done")
start_t=time.time()

for i in range(update_times):
    if i%10==0:
        print(i)
        sys.stdout.flush()
    coder = SparseCoder(dictionary=D.T,transform_n_nonzero_coefs=transform_n_nonzero_coefs, transform_algorithm="omp")
    if i==0:
        Y_init=image_vecs
        # the_H=np.zeros((n_classes,Y_init.shape[1]),dtype=int)
        X_single =np.eye(D.shape[1]) #X_single的每个列向量是一个图像的稀疏表征
        Bs=np.dot(Y_init,X_single.T)
        Cs=np.linalg.inv(np.dot(X_single,X_single.T))
    the_B=Bs
    the_C=Cs
    im_vec=image_vecs[:,i]
    new_y=np.array(im_vec,dtype = float)
    new_y=new_y.reshape(n_features,1)
    new_x=(coder.transform(new_y.T)).T
    # new_x=transform(D,new_y,transform_n_nonzero_coefs)
    new_B=the_B+np.dot(new_y,new_x.T)
    new_C=the_C-(np.matrix(the_C)*np.matrix(new_x)*np.matrix(new_x.T)*np.matrix(the_C))/(np.matrix(new_x.T)*np.matrix(the_C)*np.matrix(new_x)+1) #matrix inversion lemma(Woodbury matrix identity)
    Bs=new_B
    Cs=new_C
    new_D=np.dot(new_B,new_C)
    D=np.copy(new_D)
    Ds=D
    Ds=preprocessing.normalize(Ds.T, norm='l2').T
end_t=time.time()
print("train_time : "+str(end_t-start_t))
sys.stdout.flush()
D_all=Ds
np.save("model/D_only_"+py_file_name+"_"+str(w)+"_"+str(h)+'_'+str(transform_n_nonzero_coefs)+'_'+str(start_init_number)+"_"+str(train_number)+"_"+str(update_times),D_all)
print("D_all saved")