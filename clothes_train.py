import scipy.io
import time
import pdb
import numpy as np
from SSDL_GU import *
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
import copy
from learning_incoherent_dictionary import *
from numpy import random

for a2 in range(100):

    # data = scipy.io.loadmat('clothes5.mat') # 读取mat文件
    data = scipy.io.loadmat('T4.mat') # 读取mat文件
    # print(data.keys())  # 查看mat文件中的所有变量
    image_vecs=data['train_data']
    labels_mat=data['train_Annotation']
    labels_mat=labels_mat*2-1
    change_num=1e-7
    image_vecs=preprocessing.normalize(image_vecs.T, norm='l2').T
    image_vecs=norm_Ys(image_vecs)
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
    update_times=1000
    im_vec_len=w*h
    n_atoms = start_init_number*1
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
    D=np.empty((im_vec_len,n_atoms))
    for class_index in range(n_classes):
        D[:,:start_init_number] = image_vecs[:,inds_of_file_path[class_index][:start_init_number]]
        # D=random.random(size=(D.shape[0],D.shape[1]))
        D = norm_cols_plus_petit_1(D,c)
        Ds[class_index]=np.copy(D)

    print("initializing classifier ... done")
    start_t=time.time()

    a1=100
    # a2=80
    for i in range(update_times):
        if i%a1==a2:
            #无 0.6254180602006689
            #80 0.6454849498327759
            #a2 10 start_change 0.6622073578595318
            D_all=Ds
            D_all=D_all.transpose((0,2,1))
            D_all=D_all.reshape(-1,im_vec_len).T
            D_all=preprocessing.normalize(D_all.T, norm='l2').T
            coder = SparseCoder(dictionary=D_all.T,transform_n_nonzero_coefs=transform_n_nonzero_coefs, transform_algorithm="omp")
            the_X=(coder.transform(image_vecs.T)).T
            D_all=incoherent(D_all,image_vecs,the_X,1)
            for class_index in range(n_classes):
                Ds[class_index]=D_all[:,n_atoms*class_index:n_atoms*(class_index+1)]
        for class_index in range(n_classes):
            j_label=ind_to_lab_dir[class_index]
            if class_index==0 and i%10==0:
                print(i)
                sys.stdout.flush()
            D=Ds[class_index]
            coder = SparseCoder(dictionary=D.T,transform_n_nonzero_coefs=transform_n_nonzero_coefs, transform_algorithm="omp")
            if i==0:
                img_init_indexs=inds_of_file_path[class_index][:start_init_number]
                Y_init=image_vecs[:,img_init_indexs]
                # the_H=np.zeros((n_classes,Y_init.shape[1]),dtype=int)
                the_Q=np.zeros((n_atoms*n_classes,Y_init.shape[1]),dtype=int)
                for k in range(Y_init.shape[1]):
                    # the_H[class_index,k]=1
                    the_Q[n_atoms*class_index:n_atoms*(class_index+1),k]=1
                the_H=labels_mat[:,img_init_indexs]
                X_single =np.eye(D.shape[1]) #X_single的每个列向量是一个图像的稀疏表征
                Bs[class_index]=np.dot(Y_init,X_single.T)
                H_Bs[class_index]=np.dot(the_H,X_single.T)
                Q_Bs[class_index]=np.dot(the_Q,X_single.T)
                Cs[class_index]=np.linalg.inv(np.dot(X_single,X_single.T))
                Ws[class_index]=np.dot(H_Bs[class_index],Cs[class_index])
                As[class_index]=np.dot(Q_Bs[class_index],Cs[class_index])
            the_B=Bs[class_index]
            the_H_B=H_Bs[class_index]
            the_Q_B=Q_Bs[class_index]
            the_C=Cs[class_index]
            label_indexs_for_update=inds_of_file_path[class_index][:images_count[class_index]]
            new_index=[label_indexs_for_update[(i+start_init_number)%images_count[class_index]]]
            im_vec=image_vecs[:,new_index]
            new_y=np.array(im_vec,dtype = float)
            new_y=preprocessing.normalize(new_y.T, norm="l2").T
            new_y=new_y.reshape(n_features,1)
            new_label=class_index
            new_h=np.zeros((n_classes,1))
            lab_index=lab_to_ind_dir[new_label]
            new_h=labels_mat[:,new_index]
            new_q=np.zeros((n_atoms*n_classes,1))
            new_q[n_atoms*lab_index:n_atoms*(lab_index+1),0]=1
            new_x=(coder.transform(new_y.T)).T
            # new_x=transform(D,new_y,transform_n_nonzero_coefs)
            new_B=the_B+np.dot(new_y,new_x.T)
            new_H_B=the_H_B+np.dot(new_h,new_x.T)
            new_Q_B=the_Q_B+np.dot(new_q,new_x.T)
            new_C=the_C-(np.matrix(the_C)*np.matrix(new_x)*np.matrix(new_x.T)*np.matrix(the_C))/(np.matrix(new_x.T)*np.matrix(the_C)*np.matrix(new_x)+1) #matrix inversion lemma(Woodbury matrix identity)
            Bs[class_index]=new_B
            H_Bs[class_index]=new_H_B
            Q_Bs[class_index]=new_Q_B
            Cs[class_index]=new_C
            new_D=np.dot(new_B,new_C)
            D=np.copy(new_D)
            D=preprocessing.normalize(D.T, norm='l2').T
            Ds[class_index]=D
            Ws[class_index]=np.dot(new_H_B,new_C)
            As[class_index]=np.dot(new_Q_B,new_C)
    end_t=time.time()
    print("train_time : "+str(end_t-start_t))
    sys.stdout.flush()
    D_all=Ds
    D_all=D_all.transpose((0,2,1))
    D_all=D_all.reshape(-1,im_vec_len).T
    D_all=preprocessing.normalize(D_all.T, norm='l2').T
    coder = SparseCoder(dictionary=D_all.T,transform_n_nonzero_coefs=transform_n_nonzero_coefs, transform_algorithm="omp")
    the_X=(coder.transform(image_vecs.T)).T
    # D_all=incoherent(D_all,image_vecs,the_X,1)
    D_all=preprocessing.normalize(D_all.T, norm='l2').T
    np.save("D_all_"+py_file_name+"_mulD_"+str(w)+"_"+str(h)+'_'+str(transform_n_nonzero_coefs)+'_'+str(start_init_number)+"_"+str(train_number)+"_"+str(update_times)+"_"+str(a2),D_all)
    print("D_all saved")
    W_all=Ws
    W_all=W_all.transpose((0,2,1))
    W_all=W_all.reshape(-1,n_classes).T
    np.save("W_all_"+py_file_name+"_mulD_"+str(w)+"_"+str(h)+'_'+str(transform_n_nonzero_coefs)+'_'+str(start_init_number)+"_"+str(train_number)+"_"+str(update_times)+"_"+str(a2),W_all)
    print("W_all saved")
    A_all=As
    A_all=A_all.transpose((0,2,1))
    A_all=A_all.reshape(-1,n_classes*n_atoms).T
    np.save("A_all_"+py_file_name+"_mulD_"+str(w)+"_"+str(h)+'_'+str(transform_n_nonzero_coefs)+'_'+str(start_init_number)+"_"+str(train_number)+"_"+str(update_times)+"_"+str(a2),A_all)
    print("A_all saved")

    np.save(inds_of_file_path_path,inds_of_file_path)