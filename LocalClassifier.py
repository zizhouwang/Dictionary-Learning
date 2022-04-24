import time
import pdb
import numpy as np
from SSDL_GU import *
from li2nsvm_multiclass_lbfgs import *
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
from sklearn.decomposition import SparseCoder

class Params:
    pass

def soft(x,tau):
    temp=abs(x)-tau
    temp[temp<0]=0
    y=np.sign(x)*temp
    return y

def Coverage(Outputs,test_target):
    num_class=Outputs.shape[0]
    num_instance=Outputs.shape[1]
    Label=np.empty(num_instance,dtype=object)
    not_Label=np.empty(num_instance,dtype=object)
    for i in range(num_class):
        Label[i]=np.array([])
        not_Label[i]=np.array([])
    Label_size=np.zeros(num_instance)
    for i in range(num_instance):
        temp=test_target[:,i]
        Label_size[i]=np.sum(temp==np.ones(num_class))
        for j in range(num_class):
            if temp[j]==1:
                Label[i]=np.hstack((Label[i],np.array([j])))
            else:
                not_Label[i]=np.hstack((not_Label[i],np.array([j])))
    cover=0
    for i in range(num_instance):
        temp=Outputs[:,i]
        index=temp.argsort()
        temp.sort()
        tempvalue=temp
        temp_min=num_class+1
        for m in range(Label_size[i]):
            res=np.where(index==Label[i][m])[0]
            if res.shape[0]>0:
                if res[0]<temp_min:
                    temp_min=res[0]
        cover=cover+(num_class-temp_min+1)
    Coverage=(cover/num_instance)-1
    return Coverage

def Ranking_loss(Outputs,test_target):
    num_class=Outputs.shape[0]
    num_instance=Outputs.shape[1]
    temp_Outputs=np.array([]).reshape((num_class,0))
    temp_test_target=np.array([]).reshape((test_target.shape[0],0))
    for i in range(num_instance):
        temp=test_target[:,i]
        if (np.sum(temp)!=num_class)&(np.sum(temp)!=-num_class):
            temp_Outputs=np.hstack((temp_Outputs,Outputs[:,i].reshape(-1,1)))
            temp_test_target=np.hstack((temp_test_target,temp.reshape(-1,1)))
    Outputs = temp_Outputs
    test_target = temp_test_target
    num_class=Outputs.shape[0]
    num_instance=Outputs.shape[1]

    Label=np.empty(num_instance,dtype=object)
    not_Label=np.empty(num_instance,dtype=object)
    for i in range(num_instance):
        Label[i]=np.array([],dtype=int)
        not_Label[i]=np.array([],dtype=int)
    Label_size=np.zeros(num_instance,dtype=int)
    rl_binary=np.empty(num_instance)
    for i in range(num_instance):
        temp=test_target[:,i]
        Label_size[i]=np.sum(temp==np.ones(num_class))
        for j in range(num_class):
            if temp[j]==1:
                Label[i]=np.hstack((Label[i],np.array([j])))
            else:
                not_Label[i]=np.hstack((not_Label[i],np.array([j])))
    rankloss=0
    for i in range(num_instance):
        temp=0
        for m in range(Label_size[i]):
            for n in range(num_class-Label_size[i]):
                if Outputs[Label[i][m],i]<=Outputs[not_Label[i][n],i]:
                    temp=temp+1
        rl_binary[i]=temp/((m+1)*(n+1))
        rankloss=rankloss+temp/((m+1)*(n+1))
    RankingLoss=rankloss/num_instance
    return RankingLoss

def One_error(Outputs,test_target):
    pass

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
    Average_Precision1=correct_num*1./num_instance
    return Average_Precision,Average_Precision1

def Coef_Update_Test(X,D,A,par):
    par.nIter    =     200;
    par.isshow   =     False;
    par.citeT    =     1e-5;
    par.cT       =     1e+10;
    nIter        =    par.nIter;
    c            =    par.c;
    sigma        =    c;
    lambda1      =    par.lambda1;
    lambda2      =    par.lambda2; 
    A_mean       =    par.A_mean;
    B            =    repmat(A_mean.reshape((-1,1)),1,X.shape[1])

    #TWIST parameter
    for_ever           =         1;
    IST_iters          =         0;
    TwIST_iters        =         0;
    sparse             =         1;
    enforceMonotone    =         1;
    lam1               =         1e-4;  
    lamN               =         1;  
    rho0               =         (1-lam1/lamN)/(1+lam1/lamN)
    alpha              =         2/(1+np.sqrt(1-rho0**2))    
    beta               =         alpha*2/(lam1+lamN);       




    #main loop
    am2       =      A;
    am1       =      A;

    gap  =  norm((X-D@A),'fro')**2+2*lambda1*np.sum(abs(A.reshape(-1,1)))+lambda2*norm((A-B),'fro')**2;
    prev_f   =   gap;
    ert=[]
    ert.append(gap);
    for n_it in range(1,nIter,1):
        while for_ever:
            tem1=2*(D.T@D)@am1-2*D.T@X
            grad1=tem1.reshape((-1,1))
            tem2=2*lambda2*(am1-B)
            grad2=tem2.reshape(-1,1)
            grad=grad1+grad2
            v=am1.reshape(-1,1)-grad/(2*sigma)
            tem=soft(v,lambda1/sigma)
            a_temp=tem.reshape(D.shape[1],am1.shape[1])
            if IST_iters>=2 or TwIST_iters!=0:
                if sparse:
                    mask=(a_temp!=0)
                    am1=am1*mask
                    am2=am2*mask
                am2=(alpha-beta)*am1+(1-alpha)*am2+beta*a_temp
                gap=norm(X-D@am2)**2+2*lambda1*np.sum(abs(am2.reshape(-1,1)))+lambda2*norm(am2-B)**2
                f=gap
                if f>prev_f and enforceMonotone:
                    TwIST_iters=0
                else:
                    TwIST_iters=TwIST_iters+1
                    IST_iters=0
                    a_temp=am2
                    if TwIST_iters%10000==0:
                        c=0.9*c
                        sigma=c
                    break
            else:
                gap=norm(X-D@a_temp)**2+2*lambda1*np.sum(abs(a_temp.reshape(-1,1)))+lambda2*norm(a_temp-B)**2
                f=gap
                if f>prev_f:
                    c=2*c
                    sigma=c
                    if c>par.cT:
                        break
                    IST_iters=0
                    TwIST_iters=0
                else:
                    TwIST_iters+=1
                    break
        citerion=abs(f-prev_f)/abs(prev_f)
        if citerion<par.citeT or c>par.cT:
            break
        am2=copy.deepcopy(am1)
        am1=copy.deepcopy(a_temp)
        At_now=copy.deepcopy(a_temp)
        prev_f=copy.deepcopy(f)
        ert.append(f)
    opts=Params()
    opts.A=copy.deepcopy(At_now)
    opts.ert=copy.deepcopy(ert)
    return opts

def SparseRepresentation(X,D,A_mean,param,l1,transform_n_nonzero_coefs):
    if len(D.shape)>2:
        featureDim=D.shape[1]
        D=D.transpose(1,0,2).reshape((featureDim,-1))
    # par = Params()
    # par.lambda1 = param.lambda1;
    # par.lambda2 = param.lambda2;
    # D=np.hstack((D[0],D[1],D[2],D[3],D[4]))
    # A_ini = np.ones((transform_n_nonzero_coefs, X.shape[1]))
    # par.A_mean = A_mean[0]
    # if D.shape[0] >= D.shape[1]:
    #     w, v = LA.eig(D.T @ D)
    #     par.c = 1.05 * w.max()
    # else:
    #     w, v = LA.eig(D @ D.T)
    #     par.c = 1.05 * w.max()
    # opts = Coef_Update_Test(X, D, A_ini, par)
    # A_test = opts.A
    # P = LA.inv(D.T @ D + l1 * np.eye(D.shape[1])) @ D.T
    # A_test = P @ X
    coder = SparseCoder(dictionary=D.T, transform_n_nonzero_coefs=transform_n_nonzero_coefs,
                        transform_algorithm="omp")
    A_test = (coder.transform(X.T)).T
    A_test_nonzero=A_test
    # A_test_nonzero=np.empty((transform_n_nonzero_coefs, X.shape[1]))
    # for i in range(A_test.shape[1]):
    #     A_test_nonzero[:,i] = A_test[:,i][A_test[:,i] != 0]
    return A_test_nonzero

def LocalClassifier(X,D,A_mean,param,Uk,bk,l1,is_return_A_test=False):
    par=Params()
    par.lambda1   = param.lambda1;
    par.lambda2   = param.lambda2;
    labelNum      = D.shape[0]
    featureDim    = X.shape[0]
    testNum       = X.shape[1]
    A_test        = np.empty(5,dtype=object)
    rebuild_all   = np.zeros((labelNum,testNum));
    rebuild_nw    = np.zeros((labelNum,testNum));

    rebuild_test = np.zeros((labelNum,testNum));
    for k in range(labelNum):
        A_ini=np.ones((D[k].shape[1],X.shape[1]))
        par.A_mean=A_mean[k]
        if D[k].shape[0]>D[k].shape[1]:
            w,v=LA.eig(D[k].T@D[k])
            par.c=1.05*w.max()
        else:
            w,v=LA.eig(D[k]@D[k].T)
            par.c=1.05*w.max()
        opts=Coef_Update_Test(X,D[k],A_ini,par)
        A_test[k]=opts.A
        P=LA.inv(D[k].T@D[k]+l1*np.eye(D[k].shape[1]))@D[k].T
        A_test[k]=P@X
        rebuild_l2=(X-D[k]@A_test[k])**2
        sparse_fac=abs(A_test[k])
        within_fac=(A_test[k]-repmat(A_mean[k].reshape((-1,1)),1,testNum))**2
        s=(Uk[:,k].T@A_test[k])-bk[k]
        rebuild_all[k,:]=np.sum(rebuild_l2,axis=0)+2*par.lambda1*np.sum(sparse_fac,axis=0)-par.lambda2*(1/(1.+np.exp(-s))-0.5)
        rebuild_nw[k,:]=np.sum(rebuild_l2,axis=0)
        rebuild_test[k,:]=np.sum(rebuild_l2,axis=0)
    output1=repmat(np.max(rebuild_all,axis=0),labelNum,1)-rebuild_all
    output2=repmat(np.max(rebuild_nw,axis=0),labelNum,1)-rebuild_nw
    output3=rebuild_test
    A_test_one=np.empty(A_test[0].shape)
    for i in range(X.shape[1]):
        A_test_one[:,i]=A_test[output1[:, i].argmax()][:,i]
    # A_test[output1[:, 0].argmax()]
    if is_return_A_test:
        return output1,output2,output3,A_test_one
    return output1,output2,output3