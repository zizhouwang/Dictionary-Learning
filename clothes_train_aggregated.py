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

for a2 in range(1):
    data = scipy.io.loadmat('T4.mat') # 读取mat文件
    # print(data.keys())  # 查看mat文件中的所有变量
    image_vecs=data['train_data']
    labels_mat=data['train_Annotation']
    labels_mat=labels_mat*2-1
    change_num=1e-7
    image_vecs=preprocessing.normalize(image_vecs.T, norm='l2').T
    # image_vecs=norm_Ys(image_vecs)
    images_count=np.empty((5),dtype=int)
    for i in range(labels_mat.shape[0]):
        one_label_mat=labels_mat[i]
        one_labels_index=np.where(one_label_mat==1)[0]
        images_count[i]=one_labels_index.shape[0]
    # for i in range(labels_mat.shape[1]):
    #     new_label=np.where(labels_mat[:, i] == 1)[0][-1]
    #     labels_mat[:,i]=0
    #     labels_mat[new_label,i]=1
    t=time.time()

    np.random.seed(int(t)%100)
    n_classes=labels_mat.shape[0]
    classes=np.arange(n_classes)
    w=54
    h=46

    py_file_name="clothes"

    start_init_number=30
    train_number=300
    update_times=300
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
    n_data = image_vecs.shape[1]

    """ Start the process, initialize dictionary """
    Ds=np.empty((n_classes,im_vec_len,n_atoms))
    Ws=np.empty((n_classes,n_atoms))
    As=np.empty((n_atoms*n_classes,n_atoms))
    Bs=np.empty((im_vec_len,n_atoms))
    H_Bs=np.empty((n_classes,n_atoms))
    Q_Bs=np.empty((n_atoms*n_classes,n_atoms))
    Cs=np.empty((n_atoms,n_atoms))
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
    a1=100
    # a2=80
    for i in range(update_times):
        if i%a1==a2:
            #无 0.6521
            #a2 10 start_change 0.6622073578595318
            # D_all=Ds
            # D_all=D_all.transpose((0,2,1))
            # D_all=D_all.reshape(-1,im_vec_len).T
            # D_all=preprocessing.normalize(D_all.T, norm='l2').T
            # coder = SparseCoder(dictionary=D_all.T,transform_n_nonzero_coefs=transform_n_nonzero_coefs, transform_algorithm="omp")
            # the_X=(coder.transform(image_vecs.T)).T
            # D_all=incoherent(D_all,image_vecs,the_X,1)
            # for class_index in range(n_classes):
            #     Ds[class_index]=D_all[:,n_atoms*class_index:n_atoms*(class_index+1)]
            pass
        if i%10==0:
            print(i)
            sys.stdout.flush()
        coder = SparseCoder(dictionary=D.T,transform_n_nonzero_coefs=transform_n_nonzero_coefs, transform_algorithm="omp")
        if i==0:
            Y_indexs=np.arange(image_vecs.shape[1])
            random.shuffle(Y_indexs)
            Y_indexs=Y_indexs[:n_atoms]
            Y_init=image_vecs[:,Y_indexs]
            the_H=labels_mat[:,Y_indexs]
            # the_H=np.zeros((n_classes,Y_init.shape[1]),dtype=int)
            the_Q=np.zeros((n_atoms*n_classes,Y_init.shape[1]),dtype=int)
            for k in range(Y_init.shape[1]):
                new_label=np.where(labels_mat[:, k] == 1)[0][0]
                the_Q[n_atoms*new_label:n_atoms*(new_label+1),k]=1
            X_single =np.eye(Y_init.shape[1]) #X_single的每个列向量是一个图像的稀疏表征
            # X_single=(coder.transform(Y_init.T)).T
            Bs=np.dot(Y_init,X_single.T)
            H_Bs=np.dot(the_H,X_single.T)
            Q_Bs=np.dot(the_Q,X_single.T)
            Cs=np.linalg.inv(np.dot(X_single,X_single.T))
            Ws=np.dot(H_Bs,Cs)
            As=np.dot(Q_Bs,Cs)
        new_label=np.where(labels_mat[:, i%n_data] == 1)[0][0]
        the_B=Bs
        the_H_B=H_Bs
        the_Q_B=Q_Bs
        the_C=Cs
        im_vec=image_vecs[:,i%n_data]
        new_y=np.array(im_vec,dtype = float)
        new_y=new_y.reshape(n_features,1)
        new_h=np.zeros((n_classes,1))
        new_h[:,0]=labels_mat[:,i%n_data]
        new_q=np.zeros((n_atoms*n_classes,1))
        new_q[n_atoms*new_label:n_atoms*(new_label+1),0]=1
        new_x=(coder.transform(new_y.T)).T
        # new_x=transform(D,new_y,transform_n_nonzero_coefs)
        new_B=the_B+np.dot(new_y,new_x.T)
        new_H_B=the_H_B+np.dot(new_h,new_x.T)
        new_Q_B=the_Q_B+np.dot(new_q,new_x.T)
        new_C=the_C-(np.matrix(the_C)*np.matrix(new_x)*np.matrix(new_x.T)*np.matrix(the_C))/(np.matrix(new_x.T)*np.matrix(the_C)*np.matrix(new_x)+1) #matrix inversion lemma(Woodbury matrix identity)
        Bs=new_B
        H_Bs=new_H_B
        Q_Bs=new_Q_B
        Cs=new_C
        new_D=np.dot(new_B,new_C)
        D=np.copy(new_D)
        Ws=np.dot(new_H_B,new_C)
        As=np.dot(new_Q_B,new_C)
        for j in range(D.shape[1]):
            Ws[:,j]=Ws[:,j]/(np.sum(D[:,j]**2))
            As[:,j]=As[:,j]/(np.sum(D[:,j]**2))
        D_diff=Ds-D
        print(abs(D_diff).max())
        print(abs(D_diff).mean())
        print()
        sys.stdout.flush()
        Ds=D
        Ds=preprocessing.normalize(Ds.T, norm='l2').T
    end_t=time.time()
    print("train_time : "+str(end_t-start_t))
    sys.stdout.flush()
    D_all=Ds
    np.save("model/D_all_"+py_file_name+"_"+str(w)+"_"+str(h)+'_'+str(transform_n_nonzero_coefs)+'_'+str(start_init_number)+"_"+str(train_number)+"_"+str(update_times)+"_"+str(a2),D_all)
    print("D_all saved")
    W_all=Ws
    # W_all=W_all.transpose((0,2,1))
    # W_all=W_all.reshape(-1,n_classes).T
    np.save("model/W_all_"+py_file_name+"_"+str(w)+"_"+str(h)+'_'+str(transform_n_nonzero_coefs)+'_'+str(start_init_number)+"_"+str(train_number)+"_"+str(update_times)+"_"+str(a2),W_all)
    print("W_all saved")
    A_all=As
    # A_all=A_all.transpose((0,2,1))
    # A_all=A_all.reshape(-1,n_classes*n_atoms).T
    np.save("model/A_all_"+py_file_name+"_"+str(w)+"_"+str(h)+'_'+str(transform_n_nonzero_coefs)+'_'+str(start_init_number)+"_"+str(train_number)+"_"+str(update_times)+"_"+str(a2),A_all)
    print("A_all saved")

    os.system("python3 clothes_test.py "+str(update_times)+" "+str(n_atoms)+" "+str(transform_n_nonzero_coefs))