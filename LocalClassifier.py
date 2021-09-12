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

class Params:
    pass

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
	B            =    repmat(A_mean,1,X.shape[1]);

	#TWIST parameter
	for_ever           =         1;
	IST_iters          =         0;
	TwIST_iters        =         0;
	sparse             =         1;
	enforceMonotone    =         1;
	lam1               =         1e-4;  
	lamN               =         1;  
	rho0               =         (1-lam1/lamN)/(1+lam1/lamN); 
	alpha              =         2/(1+np.sqrt(1-rho0^2));      
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
			grad1=tem1.reshape[-1,1]
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
				gap=norm(X-D@a_temp)**2+2*lambda1*np.sum(abs(a_temp.reshape[-1,1]))+lambda2*norm(a_temp-B)**2
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
		ert.append=copy.deepcopy(f)
	opts.A=copy.deepcopy(At_now)
	opts.ert=copy.deepcopy(ert)
	return opts

def LocalClassifier(X,D,A_mean,param,Uk,bk,l1):
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
	for k in labelNum:
		A_ini=np.ones((D[k].shape[1],X.shape[1]))
		par.A_mean=A_mean[k]
		if D[k].shape[0]>D[k].shape[1]:
			w,v=LA.eig(D[k].T@D[k])
			par.c=1.05*w.max()
		else:
			w,v=LA.eig(D[k]@D[k].T)
			par.c=1.05*w.max()
		opts=Coef_Update_Test(X,D[k],A_ini.par)
		A_test[k]=opts.A
		P=LA.inv(D[k].T@D[k]+l1*np.eye(D[k].shape[1]))@D[k].T
		A_test[k]=P@X
		rebuild_l2=(X-D[k]@A_test[k])**2
		sparse_fac=abs(A_test[k])
		within_fac=(A_test[k]-repmat(A_mean[k],1,testNum))**2
		s=(Uk[:,k].T@A_test[k])-bk[k]
		rebuild_all[k,:]=np.sum(rebuild_l2,axis=0)+2*par.lambda1*np.sum(sparse_fac,axis=0)-par.lambda2*(1/(1.+math.exp(-s))-0.5)
		rebuild_nw[k,:]=np.sum(rebuild_l2,axis=0)
		rebuild_test[k,:]=np.sum(rebuild_l2,axis=0)
	output1=repmat(np.max(rebuild_all,axis=0),labelNum,1)-rebuild_all
	output2=repmat(np.max(rebuild_nw,axis=0),labelNum,1)-rebuild_nw
	output3=rebuild_test
	return output1,output2,output3