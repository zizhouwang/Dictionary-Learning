import scipy.io
import time
import pdb
import numpy as np
from SSDL_GU import *
from MIQP import *
from sklearn.decomposition import SparseCoder
from numpy.linalg import norm
import sys
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from mnist import MNIST
from PIL import Image
import os
import cv2
import copy

def Average_precision(Outputs,test_target):
    ap_binary=[]
    num_class,num_instance=Outputs.shape
    temp_Outputs=np.array([]).reshape((num_class,0))
    temp_test_target=np.array([]).reshape((test_target.shape[0],0))
    for i in range(num_instance):
        temp=test_target[:,i]
        if np.sum(temp)!=num_class and np.sum(temp)!=-num_class:
            temp_Outputs=np.hstack((temp_Outputs,Outputs[:,i].reshape(-1,1)))
            temp_test_target=np.hstack((temp_test_target,temp.reshape(-1,1)))
    Outputs=copy.deepcopy(temp_Outputs)
    test_target=copy.deepcopy(temp_test_target)
    num_class,num_instance=Outputs.shape
    Label=np.empty(num_instance,dtype=object)
    not_Label=np.empty(num_instance,dtype=object)
    for i in range(num_instance):
        Label[i]=np.array([])
        not_Label[i]=np.array([])
    Label_size=np.zeros(num_instance)
    for i in range(num_instance):
        temp=test_target[:,i]
        Label_size[i]=np.sum(temp==np.ones((num_class)))
        for j in range(num_class):
            if temp[j]==1:
                Label[i]=np.hstack((Label[i],np.array([j])))
            else:
                not_Label[i]=np.hstack((not_Label[i],np.array([j])))
    aveprec=0
    correct_num=0
    for i in range(num_instance):
        temp=Outputs[:,i]
        index=temp.argsort()
        temp.sort()
        tempvalue=temp
        indicator=np.zeros(num_class)
        for m in range(int(Label_size[i])):
            res=np.where(index==Label[i][m])[0]
            if res[0]==num_class-1:
                correct_num=correct_num+1
            if res.shape[0]==0:
                tempvalue=0
                loc=0
            else:
                tempvalue=1
                loc=res[0]
            indicator[loc]=1
        summary=0
        for m in range(int(Label_size[i])):
            res=np.where(index==Label[i][m])[0]
            if res.shape[0]==0:
                tempvalue=0
                loc=0
            else:
                tempvalue=1
                loc=res[0]
            summary+=np.sum(indicator[loc:num_class])/(num_class-loc+0)
        ap_binary.append(summary/Label_size[i])
        aveprec+=summary/Label_size[i]
    Average_Precision=aveprec/num_instance
    print("Average_Precision:"+str(Average_Precision))
    Average_Precision1=correct_num*1./num_instance
    return Average_Precision,Average_Precision1

best_start_change=None
best_a2=None
best_accuracy=-1

for a2 in range(1):
    for start_change in range(1):
        data = scipy.io.loadmat('T4.mat') # 读取mat文件
        # print(data.keys())  # 查看mat文件中的所有变量
        image_vecs=data['test_data']
        image_vecs=preprocessing.normalize(image_vecs.T, norm='l2').T
        # image_vecs=norm_Ys(image_vecs)
        labels_mat=data['test_Annotation']
        labels_mat= 2*labels_mat-1
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
        update_times=300
        im_vec_len=w*h
        n_atoms = start_init_number
        transform_n_nonzero_coefs=45
        if len(sys.argv)>3:
            update_times=int(sys.argv[1])
            n_atoms = int(sys.argv[2])
            transform_n_nonzero_coefs=int(sys.argv[3])
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

        # inds_of_file_path_path='inds_of_file_path_wzz_'+py_file_name+'_'+str(w)+'_'+str(h)+'_'+str(update_times)+'_'+str(transform_n_nonzero_coefs)+'_'+str(start_init_number)+'.npy'
        # inds_of_file_path=np.load(inds_of_file_path_path)

        D_all=np.load("model/D_all_"+py_file_name+"_"+str(w)+"_"+str(h)+'_'+str(transform_n_nonzero_coefs)+'_'+str(start_init_number)+"_"+str(train_number)+"_"+str(update_times)+"_"+str(a2)+".npy")
        W_all=np.load("model/W_all_"+py_file_name+"_"+str(w)+"_"+str(h)+'_'+str(transform_n_nonzero_coefs)+'_'+str(start_init_number)+"_"+str(train_number)+"_"+str(update_times)+"_"+str(a2)+".npy")

        average_accuracy=0.

        Y_test=image_vecs
        test_number=Y_test.shape[1]
        X_test=np.empty((D_all.shape[1],test_number))
        coder = SparseCoder(dictionary=D_all.T,transform_n_nonzero_coefs=transform_n_nonzero_coefs, transform_algorithm="omp")
        X_test=(coder.transform(Y_test.T)).T
        # X_test=transform(D_all,Y_test,transform_n_nonzero_coefs,None)
        # X_test=miqp_ys(D_all,Y_test,transform_n_nonzero_coefs)
        # for i in range(n_classes):
        #     D=D_all[:,i*n_atoms:(i+1)*n_atoms]
        #     coder = SparseCoder(dictionary=D.T,transform_n_nonzero_coefs=transform_n_nonzero_coefs, transform_algorithm="omp")
        #     X_test_part=(coder.transform(Y_test.T)).T
        #     X_test[i*n_atoms:(i+1)*n_atoms]=X_test_part
        the_H=np.dot(W_all,X_test)
        Average_precision(the_H,labels_mat)
        # right_num=0
        # for i in range(test_number):
        #     pre=the_H[:,i].argmax()
        #     if labels_mat[pre,i]==1:
        #         right_num=right_num+1
        #     else:
        #         pass
        #         # print(the_H[:,i])
        #         # print(labels_mat[:,i])
        #         # pdb.set_trace()
        # accuracy=right_num*1./test_number
        # if accuracy>best_accuracy:
        #     best_start_change=start_change
        #     best_a2=a2
        #     best_accuracy=accuracy
        #     print('start_change : '+str(start_change))
        #     print('iter : '+str(a2))
        #     print('accuracy : '+str(right_num*1./test_number))
        #     print()
        #     sys.stdout.flush()