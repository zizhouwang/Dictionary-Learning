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

# def MLDLSI2_MUL(params,is_last_floor,A1_mean,Dusage,Uk,bk):
#     transform_n_nonzero_coefs=30
#     D=copy.deepcopy(params.D0)
#     for r in range(params.max_iter):
#         dD=np.empty(NC,dtype=object)
#         print("r="+str(r)+"\n")
#         sys.stdout.flush()
#         if r==140:
#             pass
#             #130 0.8160720921590491
#             print("Start reduce coherence")
#             D_all = np.hstack((D[0], D[1], D[2], D[3], D[4]))
#             ori_gram = D_all.T @ D_all
#             ori_gram -= np.eye(ori_gram.shape[0])
#             ori_coherent = ori_gram.max()
#             coder = SparseCoder(dictionary=D_all.T, transform_n_nonzero_coefs=transform_n_nonzero_coefs,
#                                 transform_algorithm="omp")
#             the_X = (coder.transform(params.training_data.T)).T
#             D_new_all = incoherent_3000(D_all, params.training_data, the_X, 1)
#             new_gram = D_new_all.T @ D_new_all
#             new_gram -= np.eye(new_gram.shape[0])
#             new_coherent = new_gram.max()
#             for i in range(D.shape[0]):
#                 D[i] = D_new_all[:, i * transform_n_nonzero_coefs:(i + 1) * transform_n_nonzero_coefs]
#         find=np.arange(finished.shape[0])[finished<3]
#         if find.shape[0]==0:
#             if params.mu_mode[0]<0 and (params.mu0>0 or params.xmu0>0):
#                 print("** Start adding incoherence now.")
#                 print("Iteration: "+str(r))
#                 params.mu_mode[0]=r
#                 if len(params.mu_mode)==2:
#                     params.mu_mode[1]=params.mu_mode[1]+params.mu_mode[0]
#                 finished[:]=0
#             else:
#                 break
#         else:
#             pass
#         Uk, bk, class_name = li2nsvm_multiclass_lbfgs(A1_sum.T,y, tau)
#         temp_z=None
#         for c in range(NC):
#             if r>0 and c>0:
#                 a=1
#             Dc=D[c]
#             D2=np.zeros((M,int(np.sum(K)-K[c])))
#             D2_weight=np.zeros((M,int(np.sum(K)-K[c])))
#             for c2 in range(c):
#                 D2[:,int(np.sum(K[:c2])):int(np.sum(K[:c2+1]))]=D[c2]
#                 D2_weight[:,int(np.sum(K[:c2])):int(np.sum(K[:c2+1]))]=D[c2]*1.*labelCorr[c,c2]
#             for c2 in range(c+1,NC,1):
#                 D2[:,int(np.sum(K[:c2])-K[c]):int(np.sum(K[:c2+1])-K[c])]=D[c2]
#                 D2_weight[:,int(np.sum(K[:c2])-K[c]):int(np.sum(K[:c2+1])-K[c])]=D[c2]*1.*labelCorr[c,c2]
#             if finished[c]<3 or params.xmu0>0:
#                 Xbc=Xb[:,labelsb[labelname[c],:]==1]
#                 Nc=Xbc.shape[1]
#                 old_Nc=params.remember_factor*accumulated_N[c]
#                 if r>0:
#                     l1_energy[r,c]=l1_energy[r-1,c]*old_Nc
#                 else:
#                     l1_energy[r,c]=0
#                 accumulated_N[c]=old_Nc+Nc
#                 acciter[r,c]=accumulated_N[c]
#                 mu=params.mu0*accumulated_N[c]/(K[c]*K[c])
#                 xmu=params.xmu0*accumulated_N[c]/((np.sum(K)-K[c])*K[c])
#                 if params.mu_mode[0]>=0:
#                     if len(params.mu_mode)==1:
#                         fac=(r>params.mu_mode[0])
#                     else:
#                         fac=min(1,max(0,(r-params.mu_mode[0])/(params.mu_mode[1]-params.mu_mode[0])))
#                 else:
#                     fac=0
#                 mu=fac*mu
#                 xmu=fac*xmu
#                 if r==0:
#                     A1[c]=np.ones((Dc.shape[1],Xbc.shape[1]))
#                 if c>=len(P):
#                     temp_gram=D[c].T@D[c]
#                     # temp_gram[abs(temp_gram)<1e-10]=0
#                     # temp_gram[abs(temp_gram)<1e-10]=0
#                     P.append(LA.inv(temp_gram+params.model.lambda1*np.eye(D[c].T.shape[0])))
#                 else:
#                     P[c]=LA.inv(np.dot(D[c].T,D[c])+params.model.lambda1*np.eye(D[c].T.shape[0]))
#                 if r!=0:
#                     num=0
#                     if c!=0:
#                         for k in range(c):
#                             num=num+DataNum[k]
#                     for j in range(int(DataNum[c])):
#                         num=int(num)
#                         Y_labelki=np.dot(A1_sum[:,int(num)].T,Uk)+bk
#                         found_arr=Y_labelki*Y_label[num,:]
#                         loss_idx=np.arange(found_arr.shape[0])[found_arr<1]
#                         if loss_idx.shape[0]==0:
#                             A1_sum[:,num]=np.dot(np.dot(P[c],D[c].T),DataXb[:,num])
#                         else:
#                             Yi_idx=Y_label[num,loss_idx]
#                             Uk_idx=Uk[:,loss_idx]
#                             bk_idx=bk[loss_idx]
#                             ski=np.dot(D[c].T,DataXb[:,num])+2*lambda2*theta*(np.dot(Uk_idx,Yi_idx.T)-np.dot(Uk_idx,bk_idx.T))
#                             Tki=LA.inv(np.eye(Uk_idx.shape[1])+2*lambda2*theta*np.dot(np.dot(Uk_idx.T,P[c]),Uk_idx))
#                             # P[c]=np.around(P[c], decimals=4)
#                             A1_sum[:,num]=(P[c]-2*lambda2*theta*P[c]@Uk_idx@Tki@Uk_idx.T@P[c])@ski
#                         num+=1
#                     num=0
#                     if c!=0:
#                         for k in range(c):
#                             num=num+DataNum[k]
#                     if r>0 and c==0:
#                         a=1
#                     if r>0 and c>0:
#                         a=1
#                     for j in range(int(DataNum[c])):
#                         A1[c][:,j]=A1_sum[:,num]
#                         num+=1
#                     A1_mean[c]=np.mean(A1[c],axis=1)
#                     temp_z=copy.deepcopy(A1_sum)
#                 else:
#                     A1[c]=P[c]@D[c].T@X[c]
#                     if temp_z is None:
#                         temp_z=copy.deepcopy(A1[c])
#                     else:
#                         temp_z=np.hstack((temp_z,A1[c]))
#                     A1_mean[c]=np.mean(A1[c],axis=1)
#                 new_energy[r,c]=ModelEnergy2(Xbc,D[c],A1[c],params.model)
#                 l1_energy[r,c]=(l1_energy[r,c]+new_energy[r,c])/accumulated_N[c]
#                 XAt[c]=params.remember_factor*XAt[c]+Xbc@A1[c].T
#                 AAt[c]=params.remember_factor*AAt[c]+A1[c]@A1[c].T
#                 batch_usage=np.sum(A1[c]!=0,axis=1)
#                 Dusage[c]=params.remember_factor*Dusage[c]+batch_usage
#                 Dmean[c]=params.remember_factor*Dmean[c]+(np.sum(A1[c],axis=1).T)/batch_usage
#                 Dvar[c]=params.remember_factor*Dvar[c]+(np.sum(A1[c]*A1[c],axis=1).T)/batch_usage
#                 params.dict_update.mu       = mu;
#                 params.dict_update.xmu      = xmu;
#                 params.dict_update.positive = params.positive;
#                 params.dict_update.c        = c;
#                 params.dict_update.NC       = NC;
#                 # Dc,stuck=DictUpdate(D[c],AAt[c],XAt[c],D2_weight,params.dict_update)
#                 Dc=DictUpdate(D[c],AAt[c],XAt[c],D2_weight,params.dict_update)
#                 if np.sum(Dc*Dc,axis=0).max()>(1+sqrt(np.spacing(1))):
#                     print("error np.sum(Dc*Dc,axis=0).max()>(1+sqrt(eps))")
#                     pdb.set_trace()
#                 if params.discard_unused_atoms>0:
#                     kt_usage[c]=(Dusage[c]+0.5)/(accumulated_N[c]+1)
#                     thres=params.discard_unused_atoms
#                     found_arr=kt_usage[c]
#                     dead_atoms=np.arange(found_arr.shape[0])[found_arr<thres]
#                     if dead_atoms.shape[0]>=Dc.shape[1]:
#                         print('All atoms were discarded! Consider reducing threshold.')
#                         params.discard_unused_atoms = params.discard_unused_atoms / 10
#                     aux=np.arange(Xb.shape[1])
#                     random.shuffle(aux)
#                     aux = aux[0:min(Dc.shape[1],Xb.shape[1],dead_atoms.shape[0])]
#                     if aux.shape[0]>0:
#                         Dc[:,dead_atoms[:aux.shape[0]]]=preprocessing.normalize(Xb[:,aux].T, norm='l2').T
#                     Dusage[c][dead_atoms]=Nc/2.
#                     if dead_atoms.shape[0]>0:
#                         print('reset '+str(dead_atoms.shape[0]+' atoms:'))
#                         #???原版本这里一堆输出 别的啥也没有
#                 dD[c]=Dc-D[c]
#                 dDn[c]=LA.norm(dD[c][:])
#                 D[c]=Dc
#                 mc=abs(Dc.T@Dc)
#                 mc=mc-np.diag(np.diag(mc))
#                 mc=mc.reshape(-1,1)
#                 max_coherence[r,c]=mc.max()
#                 mean_coherence[r,c]=np.mean(mc,axis=0)
#             else:
#                 l1_energy[r,c]=l1_energy[r-1,c]
#                 cost[r,c]=cost[r-1,c]
#                 max_coherence[r,c]=max_coherence[r-1,c]
#                 mean_coherence[r,c]=mean_coherence[r-1,c]
#             xc[c]=abs(D2_weight.T@Dc)
#             max_cross_coherence[r,c]=np.mean(xc[c].reshape((-1,1)))
#             cost[r,c]=l1_energy[r,c]
#             if params.mu0>0:
#                 mu=params.mu0/(K[c]*K[c])
#                 Dc_gram=Dc.T@Dc
#                 Dc_gram_2=Dc_gram*Dc_gram
#                 cost[r,c]=cost[r,c]+mu*np.sum(np.sum(Dc_gram_2,axis=0))
#             if params.xmu0>0:
#                 xc_2=xc[c]*xc[c]
#                 cost[r,c]=cost[r,c]+xmu*np.sum(np.sum(xc_2,axis=0))
#             if r>0:
#                 cost_dif=abs(cost[r-1,c]-cost[r,c])/cost[0,c]#原版本说这里是否应修改
#                 print("cost_dif:"+str(cost_dif))
#                 if cost_dif<params.min_change:
#                     finished[c]=finished[c]+1
#                 else:
#                     finished[c]=0
#         A1_sum=copy.deepcopy(temp_z)
#         # print(abs(temp_z).sum())
#         pass
#         #save('xin','Uk','D','AAt','XAt','Dusage','xc','mean_coherence','max_coherence','mean_cross_coherence','max_cross_coherence','finished','r','l1_energy','new_energy','acciter','cost');
#         #保存为xin.mat

def MLDLSI2(params):#[D,A1_mean,Dusage,Uk,bk]
    transform_n_nonzero_coefs=30
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
    DataXb=np.empty((Xb.shape[0],0))
    for i in range(NC):
        X[i]=copy.deepcopy(Xb[:,labelsb[i,:]==1])
        DataNum[i]=X[i].shape[1]
        DataXb=np.hstack((DataXb,X[i]))
    Uinit=np.zeros((DataXb.shape[0],NC))
    binit=np.zeros(NC)
    labelsb = labelsb*2-1
    NumLabels=int(np.sum(np.sum(labelsb,axis=0).T))
    A1_sum=np.ones((30,NumLabels))
    A1_ini=np.ones((30,NumLabels))
    Uk = Uinit
    bk = binit
    lambda1  = 2e-3
    lambda2 = 0.006
    theta      =  8
    tau = 1/theta

    y=np.zeros(NumLabels)
    for i in range(NC):
        num=0
        if i!=0:
            for k in range(i):
                num=num+DataNum[k]
        for j in range(int(DataNum[i])):
            num=int(num)
            y[num]=i
            num+=1
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

    D=copy.deepcopy(params.D0)

    Uk=np.empty((D[0].shape[1],NC),dtype='float64')
    bk=np.empty(NC,dtype='float64')
    def train_one_time(r):
    # for r in range(params.max_iter):
        dD=np.empty(NC,dtype=object)
        print("r="+str(r)+"\n")
        sys.stdout.flush()
        if r==140:
            pass
            #130 0.8160720921590491
            print("Start reduce coherence")
            D_all = np.hstack((D[0], D[1], D[2], D[3], D[4]))
            ori_gram = D_all.T @ D_all
            ori_gram -= np.eye(ori_gram.shape[0])
            ori_coherent = ori_gram.max()
            coder = SparseCoder(dictionary=D_all.T, transform_n_nonzero_coefs=transform_n_nonzero_coefs,
                                transform_algorithm="omp")
            the_X = (coder.transform(params.training_data.T)).T
            D_new_all = incoherent_3000(D_all, params.training_data, the_X, 1)
            new_gram = D_new_all.T @ D_new_all
            new_gram -= np.eye(new_gram.shape[0])
            new_coherent = new_gram.max()
            for i in range(D.shape[0]):
                D[i] = D_new_all[:, i * transform_n_nonzero_coefs:(i + 1) * transform_n_nonzero_coefs]
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
                return D,A1_mean,Dusage,Uk,bk
        else:
            pass
        Uk[:], bk[:], class_name = li2nsvm_multiclass_lbfgs(A1_sum.T,y, tau)
        temp_z=None
        for c in range(NC):
            if r>0 and c>0:
                a=1
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
                Xbc=Xb[:,labelsb[labelname[c],:]==1]
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
                    if r>0 and c==0:
                        a=1
                    if r>0 and c>0:
                        a=1
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
                if params.discard_unused_atoms>0:
                    kt_usage[c]=(Dusage[c]+0.5)/(accumulated_N[c]+1)
                    thres=params.discard_unused_atoms
                    found_arr=kt_usage[c]
                    dead_atoms=np.arange(found_arr.shape[0])[found_arr<thres]
                    if dead_atoms.shape[0]>=Dc.shape[1]:
                        print('All atoms were discarded! Consider reducing threshold.')
                        params.discard_unused_atoms = params.discard_unused_atoms / 10
                    aux=np.arange(Xb.shape[1])
                    random.shuffle(aux)
                    aux = aux[0:min(Dc.shape[1],Xb.shape[1],dead_atoms.shape[0])]
                    if aux.shape[0]>0:
                        Dc[:,dead_atoms[:aux.shape[0]]]=preprocessing.normalize(Xb[:,aux].T, norm='l2').T
                    Dusage[c][dead_atoms]=Nc/2.
                    if dead_atoms.shape[0]>0:
                        print('reset '+str(dead_atoms.shape[0]+' atoms:'))
                        #???原版本这里一堆输出 别的啥也没有
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
                print("cost_dif:"+str(cost_dif))
                if cost_dif<params.min_change:
                    finished[c]=finished[c]+1
                else:
                    finished[c]=0
        A1_sum[:]=copy.deepcopy(temp_z)
        # A1_sum=copy.deepcopy(temp_z)
        pass
        #save('xin','Uk','D','AAt','XAt','Dusage','xc','mean_coherence','max_coherence','mean_cross_coherence','max_cross_coherence','finished','r','l1_energy','new_energy','acciter','cost');
        #保存为xin.mat
    return train_one_time()