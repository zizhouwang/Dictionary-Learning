import time
import pdb
import numpy as np
from SSDL_GU import *
from DictUpdate import *
from li2nsvm_multiclass_lbfgs import *
from learning_incoherent_dictionary import *
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
from scipy.spatial.distance import pdist, squareform
import copy
import scipy.io as scio
import random
from numpy.matlib import repmat

def ModelEnergy2(X,D,A,params):
    if params.reg_mode==1:
        if params.reg_type=="l1":
            pass
            print("error function ModelEnergy2 params.reg_mode 1")
            pdb.set_trace()
            #懒得写了
    elif params.reg_mode==0:
        print("error function ModelEnergy2 params.reg_mode 0")
        pdb.set_trace()
        pass
    elif params.reg_mode==2:
        D.dtype="float"
        E=X-D@A
        E=LA.norm(E)
        E=E*E*0.5
        E_within=A-repmat(np.mean(A,axis=1).reshape((-1,1)),1,A.shape[1])
        E_within=LA.norm(E_within)
        E_within=E_within*E_within
        E_within=E_within*params.lambda2*0.5
        E=E+E_within
        if params.reg_type=="l1":
            R=E+params.the_lambda*np.sum(abs(A[:]))
            pass
        else:
            print("error function ModelEnergy2 params.reg_type!=l1")
            pdb.set_trace()
    else:
        print("error function ModelEnergy2 params.reg_mode")
        pdb.set_trace()
    return R

def GetPrefix(params):
    path = 'C:/Users/zxzas/Desktop/Dictionary-Learning/model/'
    path = './model/'
    prefix = path+'model-'+time.strftime('%Y-%m-%d-%H-%M',time.localtime(time.time()))
    return prefix

def RLSDLA(n_atoms,transform_n_nonzero_coefs):
    data = scipy.io.loadmat('T4.mat')  # 读取mat文件
    image_vecs = data['train_data']
    labels_mat = data['train_Annotation']
    n_classes = labels_mat.shape[0]
    DataXb=np.empty((image_vecs.shape[0],0))
    for i in range(n_classes):
        DataXb=np.hstack((DataXb,copy.deepcopy(image_vecs[:,labels_mat[i,:]==1])))
    image_vecs=DataXb
    image_vecs = preprocessing.normalize(image_vecs.T, norm='l2').T
    # image_vecs = norm_Ys(image_vecs)
    t = time.time()

    np.random.seed(int(t) % 100)
    classes = np.arange(n_classes)
    w = 54
    h = 46

    py_file_name = "clothes"

    update_times = 10
    im_vec_len = w * h
    n_neighbor = 8
    lamda = 0.5
    beta = 1.
    gamma = 1.
    mu = 2. * gamma
    r = 2.
    c = 1.

    seed = 0  # to save the way initialize dictionary
    n_iter_sp = 50  # number max of iteration in sparse coding
    n_iter_du = 50  # number max of iteration in dictionary update
    n_iter = 15  # number max of general iteration
    n_features = image_vecs.shape[0]
    n_data = image_vecs.shape[1]

    """ Start the process, initialize dictionary """
    Ds = np.empty((n_classes, im_vec_len, n_atoms))
    Bs = np.empty((im_vec_len, n_atoms))
    Cs = np.empty((n_atoms, n_atoms))
    D = np.empty((im_vec_len, n_atoms))
    # for class_index in range(n_classes):
    #     D[:,:start_init_number] = image_vecs[:,inds_of_file_path[class_index][:start_init_number]]
    #     # D=random.random(size=(D.shape[0],D.shape[1]))
    #     D = norm_cols_plus_petit_1(D,c)
    #     Ds[class_index]=copy.deepcopy(D)
    # D=Ds.transpose((0,2,1)).reshape(-1,im_vec_len).T
    D = np.random.rand(im_vec_len, n_atoms)


    D=scipy.io.loadmat('D_random_init.mat')['D_init']


    D = preprocessing.normalize(D.T, norm='l2').T
    Ds = D
    print("initializing classifier ... done")
    start_t = time.time()

    Y_indexs = np.arange(image_vecs.shape[1])
    random.shuffle(Y_indexs)


    Y_indexs=scipy.io.loadmat('Y_indexs.mat')['Y_indexs'][0]


    Y_indexs_part = Y_indexs[:n_atoms]
    Y_init = image_vecs[:, Y_indexs_part]
    coder = SparseCoder(dictionary=D.T, transform_n_nonzero_coefs=transform_n_nonzero_coefs,
                        transform_algorithm="omp")
    X_single = np.eye(D.shape[1])
    # X_single=(coder.transform(Y_init.T)).T
    Bs = np.dot(Y_init, X_single.T)
    Cs = np.linalg.inv(np.dot(X_single, X_single.T))

    def train_one_time(i):
        if i % 10 == 0:
            # print(i)
            sys.stdout.flush()
        coder = SparseCoder(dictionary=D.T, transform_n_nonzero_coefs=transform_n_nonzero_coefs,
                            transform_algorithm="omp")
        the_B = Bs
        the_C = Cs
        im_vec = image_vecs[:, i%n_data]
        new_y = np.array(im_vec, dtype=float)
        new_y = new_y.reshape(n_features, 1)
        new_x = (coder.transform(new_y.T)).T
        # new_x=transform(D,new_y,transform_n_nonzero_coefs)
        new_B = the_B + np.dot(new_y, new_x.T)
        new_C = the_C - (np.matrix(the_C) * np.matrix(new_x) * np.matrix(new_x.T) * np.matrix(the_C)) / (
                    np.matrix(new_x.T) * np.matrix(the_C) * np.matrix(
                new_x) + 1)  # matrix inversion lemma(Woodbury matrix identity)
        Bs[:] = new_B
        Cs[:] = new_C
        new_D = np.dot(new_B, new_C)
        D_diff=new_D-D
        D[:] = copy.deepcopy(new_D)
        Ds[:] = D
        Ds[:] = preprocessing.normalize(Ds.T, norm='l2').T
        # print(abs(D_diff).max())
        # print(abs(D_diff).mean())
        # print()
        # coder = SparseCoder(dictionary=Ds.T, transform_n_nonzero_coefs=transform_n_nonzero_coefs,
        #                     transform_algorithm="omp")
        # X_all = (coder.transform(image_vecs.T)).T
        # return Ds,X_all
        return Ds
    return train_one_time,copy.deepcopy(D),copy.deepcopy(Y_indexs_part)

def MLDLSI2(params,y,atom_n):#[D,A1_mean,Dusage,Uk,bk]
    transform_n_nonzero_coefs=atom_n
    NC=params.D0.shape[0]
    K=np.zeros(NC)
    labelname=np.arange(params.training_labels.shape[0])
    labelCorr=np.ones((NC,NC))
    if params.dict_update.xcorr==1:
        for i in range(NC):
            for j in range(NC):
                labelCorr[i,j]=pdist([params.training_labels[i,:]+0.,params.training_labels[j,:]+0.],'cosine')
    for i in range(NC):
        M,K[i]=params.D0[i].shape
    Xb=copy.deepcopy(params.training_data)
    labelsb=copy.deepcopy(params.training_labels)
    D=copy.deepcopy(params.D0)
    r0=1
    dDn=np.zeros(NC)
    finished=np.zeros(NC)
    AAt=np.empty(NC,dtype=object)
    A1=np.empty(NC,dtype=object)
    A_ini=np.empty(NC,dtype=object)
    XAt=np.empty(NC,dtype=object)
    Dusage=np.empty(NC,dtype=object)
    Dmean=np.empty(NC,dtype=object)
    Dvar=np.empty(NC,dtype=object)
    kt_usage=np.empty(NC,dtype=object)
    xc=np.empty(NC,dtype=object)
    A1_mean=np.empty(NC,dtype=object)

    for i in range(NC):
        matrix_len=int(K[i])
        Dusage[i]=np.zeros(matrix_len)
        Dmean[i]=np.zeros(matrix_len)
        Dvar[i]=np.zeros(matrix_len)
        AAt[i]=np.zeros((matrix_len,matrix_len))
        XAt[i]=np.zeros((M,matrix_len))
        A1_mean[i]=np.zeros((matrix_len,1))
    cost=np.zeros((params.max_iter,NC))
    l1_energy=np.zeros((params.max_iter,NC))
    l1_energy2=np.zeros((params.max_iter,NC))
    new_energy=np.zeros((params.max_iter,NC))
    mean_coherence=np.zeros((params.max_iter,NC))
    max_coherence=np.zeros((params.max_iter,NC))
    max_cross_coherence=np.zeros((params.max_iter,NC))
    mean_cross_coherence=np.zeros((params.max_iter,NC))
    accumulated_N=np.zeros((NC))
    acciter=np.zeros((params.max_iter,NC))
    P=[]
    DataNum=np.zeros(NC,dtype=int)
    X=np.empty(NC,dtype=object)
    # DataXb=np.empty((Xb.shape[0],0))
    # for i in range(NC):
    #     X[i]=copy.deepcopy(Xb[:,labelsb[i,:]==1])
    #     DataNum[i]=X[i].shape[1]
    #     DataXb=np.hstack((DataXb,X[i]))
    Uinit=np.zeros((Xb.shape[0],NC))
    binit=np.zeros(NC)
    labelsb = labelsb*2-1
    NumLabels=int(np.sum(np.sum(labelsb,axis=0).T))
    A1_sum=np.ones((atom_n,NumLabels))
    A1_ini=np.ones((atom_n,NumLabels))
    Uk = Uinit
    bk = binit
    lambda1  = 2e-3
    lambda2 = 0.006
    theta      =  8
    tau = 1/theta

    # y=np.zeros(NumLabels)
    # for i in range(NC):
    #     num=0
    #     if i!=0:
    #         for k in range(i):
    #             num=num+DataNum[k]
    #     for j in range(int(DataNum[i])):
    #         num=int(num)
    #         y[num]=i
    #         num+=1
    class_list=np.unique(y)
    class_idx=np.zeros((NumLabels,1))
    class_idx[:]=-1
    class_space=1
    Y=np.zeros((NumLabels,NC))
    for i in range(NumLabels):
        for j in range(class_space):
            if y[i]==class_list[j]:
                class_idx[i]=j
        if class_idx[i]==-1:
            class_space=class_space+1
            class_idx[i]=class_space-1
        Y[i,int(class_idx[i])]=1
    Y_label=np.sign(Y-0.5)

    Uk=np.empty((D[0].shape[1],NC),dtype='float64')
    bk=np.empty(NC,dtype='float64')
    def train_one_time(r,is_last_layer,upper_A=None):
        if upper_A is not None:
            DataXb=upper_A
            for i in range(NC):
                DataNum[i]=np.sum(y==i)
                X[i]=DataXb[:, DataNum[0:i].sum():DataNum[0:i].sum() + DataNum[i]]
        else:
            DataXb = np.empty((Xb.shape[0], 0))
            for i in range(NC):
                X[i] = copy.deepcopy(Xb[:, labelsb[i, :] == 1])
                DataNum[i] = X[i].shape[1]
                DataXb = np.hstack((DataXb, X[i]))
        dD=np.empty(NC,dtype=object)
        print("r="+str(r)+"\n")
        sys.stdout.flush()
        # if r==140:
        #     pass
        #     #130 0.8160720921590491
        #     print("Start reduce coherence")
        #     D_all = np.hstack((D[0], D[1], D[2], D[3], D[4]))
        #     ori_gram = D_all.T @ D_all
        #     ori_gram -= np.eye(ori_gram.shape[0])
        #     ori_coherent = ori_gram.max()
        #     coder = SparseCoder(dictionary=D_all.T, transform_n_nonzero_coefs=transform_n_nonzero_coefs,
        #                         transform_algorithm="omp")
        #     the_X = (coder.transform(Xb.T)).T
        #     D_new_all = incoherent_3000(D_all, Xb, the_X, 1)
        #     new_gram = D_new_all.T @ D_new_all
        #     new_gram -= np.eye(new_gram.shape[0])
        #     new_coherent = new_gram.max()
        #     for i in range(D.shape[0]):
        #         D[i] = D_new_all[:, i * transform_n_nonzero_coefs:(i + 1) * transform_n_nonzero_coefs]
        find=np.arange(finished.shape[0])[finished<3]
        if find.shape[0]==0:
            if params.mu_mode[0]<0 and (params.mu0>0 or params.xmu0>0):
                print("** Start adding incoherence now.")
                print("Iteration: "+str(r))
                params.mu_mode[0]=r
                if len(params.mu_mode)==2:
                    params.mu_mode[1]=params.mu_mode[1]+params.mu_mode[0]
                finished[:]=0
            else:
                return D,A1_mean,Dusage,Uk,bk,A1_sum,y,True
        else:
            pass
        if is_last_layer:
            pass
        Uk[:], bk[:] = li2nsvm_multiclass_lbfgs(A1_sum.T,y, tau)
        # except AttributeError:
        #     a=1
        temp_z=None
        for c in range(NC):
            Dc=D[c]
            D2=np.zeros((M,int(np.sum(K)-K[c])))
            D2_weight=np.zeros((M,int(np.sum(K)-K[c])))
            for c2 in range(c):
                D2[:,int(np.sum(K[:c2])):int(np.sum(K[:c2+1]))]=D[c2]
                D2_weight[:,int(np.sum(K[:c2])):int(np.sum(K[:c2+1]))]=D[c2]*1.*labelCorr[c,c2]
            for c2 in range(c+1,NC,1):
                D2[:,int(np.sum(K[:c2])-K[c]):int(np.sum(K[:c2+1])-K[c])]=D[c2]
                D2_weight[:,int(np.sum(K[:c2])-K[c]):int(np.sum(K[:c2+1])-K[c])]=D[c2]*1.*labelCorr[c,c2]
            if finished[c]<3 or params.xmu0>0:
                Xbc=DataXb[:, DataNum[0:c].sum():DataNum[0:c].sum() + DataNum[c]]
                Nc=Xbc.shape[1]
                old_Nc=params.remember_factor*accumulated_N[c]
                if r>0:
                    l1_energy[r,c]=l1_energy[r-1,c]*old_Nc
                else:
                    l1_energy[r,c]=0
                accumulated_N[c]=old_Nc+Nc
                acciter[r,c]=accumulated_N[c]
                mu=params.mu0*accumulated_N[c]/(K[c]*K[c])
                xmu=params.xmu0*accumulated_N[c]/((np.sum(K)-K[c])*K[c])
                if params.mu_mode[0]>=0:
                    if len(params.mu_mode)==1:
                        fac=(r>params.mu_mode[0])
                    else:
                        fac=min(1,max(0,(r-params.mu_mode[0])/(params.mu_mode[1]-params.mu_mode[0])))
                else:
                    fac=0
                mu=fac*mu
                xmu=fac*xmu
                if r==0:
                    A1[c]=np.ones((Dc.shape[1],Xbc.shape[1]))
                if c>=len(P):
                    temp_gram=D[c].T@D[c]
                    # temp_gram[abs(temp_gram)<1e-10]=0
                    # temp_gram[abs(temp_gram)<1e-10]=0
                    P.append(LA.inv(temp_gram+params.model.lambda1*np.eye(D[c].T.shape[0])))
                else:
                    P[c]=LA.inv(np.dot(D[c].T,D[c])+params.model.lambda1*np.eye(D[c].T.shape[0]))
                if r!=0:
                    num=0
                    if c!=0:
                        for k in range(c):
                            num=num+DataNum[k]
                    for j in range(int(DataNum[c])):
                        num=int(num)
                        Y_labelki=np.dot(A1_sum[:,int(num)].T,Uk)+bk
                        found_arr=Y_labelki*Y_label[num,:]
                        loss_idx=np.arange(found_arr.shape[0])[found_arr<1]
                        if loss_idx.shape[0]==0:
                            A1_sum[:,num]=np.dot(np.dot(P[c],D[c].T),DataXb[:,num])
                        else:
                            Yi_idx=Y_label[num,loss_idx]
                            Uk_idx=Uk[:,loss_idx]
                            bk_idx=bk[loss_idx]
                            ski=np.dot(D[c].T,DataXb[:,num])+2*lambda2*theta*(np.dot(Uk_idx,Yi_idx.T)-np.dot(Uk_idx,bk_idx.T))
                            Tki=LA.inv(np.eye(Uk_idx.shape[1])+2*lambda2*theta*np.dot(np.dot(Uk_idx.T,P[c]),Uk_idx))
                            # P[c]=np.around(P[c], decimals=4)
                            A1_sum[:,num]=(P[c]-2*lambda2*theta*P[c]@Uk_idx@Tki@Uk_idx.T@P[c])@ski
                        num+=1
                    num=0
                    if c!=0:
                        for k in range(c):
                            num=num+DataNum[k]
                    for j in range(int(DataNum[c])):
                        A1[c][:,j]=A1_sum[:,num]
                        num+=1
                    A1_mean[c]=np.mean(A1[c],axis=1)
                    temp_z=copy.deepcopy(A1_sum)
                else:
                    A1[c]=P[c]@D[c].T@X[c]
                    if temp_z is None:
                        temp_z=copy.deepcopy(A1[c])
                    else:
                        temp_z=np.hstack((temp_z,A1[c]))
                    A1_mean[c]=np.mean(A1[c],axis=1)
                new_energy[r,c]=ModelEnergy2(Xbc,D[c],A1[c],params.model)
                l1_energy[r,c]=(l1_energy[r,c]+new_energy[r,c])/accumulated_N[c]
                XAt[c]=params.remember_factor*XAt[c]+Xbc@A1[c].T
                AAt[c]=params.remember_factor*AAt[c]+A1[c]@A1[c].T
                batch_usage=np.sum(A1[c]!=0,axis=1)
                Dusage[c]=params.remember_factor*Dusage[c]+batch_usage
                Dmean[c]=params.remember_factor*Dmean[c]+(np.sum(A1[c],axis=1).T)/batch_usage
                Dvar[c]=params.remember_factor*Dvar[c]+(np.sum(A1[c]*A1[c],axis=1).T)/batch_usage
                params.dict_update.mu       = mu;
                params.dict_update.xmu      = xmu;
                params.dict_update.positive = params.positive;
                params.dict_update.c        = c;
                params.dict_update.NC       = NC;
                # Dc,stuck=DictUpdate(D[c],AAt[c],XAt[c],D2_weight,params.dict_update)
                Dc=DictUpdate(D[c],AAt[c],XAt[c],D2_weight,params.dict_update)
                if np.sum(Dc*Dc,axis=0).max()>(1+sqrt(np.spacing(1))):
                    print("error np.sum(Dc*Dc,axis=0).max()>(1+sqrt(eps))")
                    pdb.set_trace()
                dD[c]=Dc-D[c]
                dDn[c]=LA.norm(dD[c][:])
                D[c]=Dc
                mc=abs(Dc.T@Dc)
                mc=mc-np.diag(np.diag(mc))
                mc=mc.reshape(-1,1)
                max_coherence[r,c]=mc.max()
                mean_coherence[r,c]=np.mean(mc,axis=0)
            else:
                l1_energy[r,c]=l1_energy[r-1,c]
                cost[r,c]=cost[r-1,c]
                max_coherence[r,c]=max_coherence[r-1,c]
                mean_coherence[r,c]=mean_coherence[r-1,c]
            xc[c]=abs(D2_weight.T@Dc)
            max_cross_coherence[r,c]=np.mean(xc[c].reshape((-1,1)))
            cost[r,c]=l1_energy[r,c]
            if params.mu0>0:
                mu=params.mu0/(K[c]*K[c])
                Dc_gram=Dc.T@Dc
                Dc_gram_2=Dc_gram*Dc_gram
                cost[r,c]=cost[r,c]+mu*np.sum(np.sum(Dc_gram_2,axis=0))
            if params.xmu0>0:
                xc_2=xc[c]*xc[c]
                cost[r,c]=cost[r,c]+xmu*np.sum(np.sum(xc_2,axis=0))
            if r>0:
                cost_dif=abs(cost[r-1,c]-cost[r,c])/cost[0,c]#原版本说这里是否应修改
                # print("cost_dif:"+str(cost_dif))
                if cost_dif<params.min_change:
                    finished[c]=finished[c]+1
                else:
                    finished[c]=0
        A1_sum[:]=copy.deepcopy(temp_z)
        # A1_sum=copy.deepcopy(temp_z)
        pass
        #save('xin','Uk','D','AAt','XAt','Dusage','xc','mean_coherence','max_coherence','mean_cross_coherence','max_cross_coherence','finished','r','l1_energy','new_energy','acciter','cost');
        #保存为xin.mat
        return D,A1_mean,Dusage,Uk,bk,A1_sum,y,False
    return train_one_time