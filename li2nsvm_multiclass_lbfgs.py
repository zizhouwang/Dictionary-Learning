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
from scipy.spatial.distance import pdist, squareform
import copy
import scipy.io as scio

class Params:
    pass

def zoom(alo, ahi, x0, f0, g0, s0, c1, c2, linegrad0, falo,galo, fhi, ghi,  f, X,Y,the_lambda,sigma,gamma):
    echo=False
    i=0
    while True:
        d1=galo+ghi-3*(falo-fhi)/(alo-ahi)
        d2=np.sqrt(d1*d1-galo*ghi)
        aj=ahi-(ahi-alo)*(ghi+d2-d1)/(ghi-galo+2*d2)
        if echo==True:
            pass
        if alo<ahi:
            if aj<alo or aj>ahi:
                aj=(alo+ahi)/2
        else:
            if aj>alo or aj<ahi:
                aj = (alo+ahi)/2
        xstar=x0+aj*s0
        fj,gj=li2nsvm_grad(xstar,X,Y,the_lambda,sigma,gamma)
        if fj>f0+c1*aj*linegrad0 or fj>falo:
            ahi=aj
            fhi=fj
            ghi=np.dot(gj.T,s0)
        else:
            linegradj=np.dot(gj.T,s0)
            if abs(linegradj) <= -c2*linegrad0:
                astar=aj
                fstar=fj
                gstar=gj
                return astar,xstar,fstar,gstar
            if linegradj*(ahi-alo)>=0:
                ahi=alo
                fhi=falo
                ghi=galo
            alo=aj
            falo=fj
            galo=linegradj
        if abs(alo-ahi)<=0.01*alo or i>=10:
            astar=aj
            fstar=fj
            gstar=gj
            if echo==True:
                pass
            return astar,xstar,fstar,gstar
        i+=1
    return astar,xstar,fstar,gstar

def lineSearchWolfe(x0, f0, g0, s0, a1, amax, c1, c2, maxiter, f, X,Y,the_lambda,sigma,gamma):
    echo=False
    ai_1=0
    ai=a1
    i=1
    fi_1=f0
    linegrad0=np.dot(g0[:].T,s0[:])
    linegradi_1=linegrad0
    while True:
        xstar=x0+ai*s0
        fi,gi=li2nsvm_grad(xstar,X,Y,the_lambda,sigma,gamma)
        linegradi=np.dot(gi[:].T,s0[;])
        if fi>(f0+c1*ai*linegrad0) or (fi>=fi_1 and i>1):
            if echo==True:
                pass
            astar,xstar,fstar,gstar=zoom(ai_1, ai,x0,f0, g0, s0, c1, c2, linegrad0, fi_1,  linegradi_1,fi, linegradi,f, X,Y,the_lambda,sigma,gamma);
            return astar,xstar,fstar,gstar
        if abs(linegradi)<=-c2*linegrad0:
            astar=ai
            fstar=fi
            gstar=gi
        if linegradi>=0:
            if echo==True:
                pass
            astar,xstar,fstar,gstar=zoom(ai, ai_1, x0,f0, g0, s0, c1, c2, linegrad0, fi, linegradi, fi_1, linegradi_1, f, X,Y,the_lambda,sigma,gamma);
            return astar,xstar,fstar,gstar
        i=i+1
        if abs(ai-amax)<=0.01*amax or i>maxiter:
            fstar=fi
            gstar=gi
            astar=ai
            if echo==True:
                pass
            returnastar,xstar,fstar,gstar
        ai_1=ai
        fi_1=fi
        linegradi_1=linegradi
        if echo==True:
            pass
        ai=(ai+amax)/2.
        if echo==True:
            pass
    return astar,xstar,fstar,gstar

def oneofc(label):
    N=label.shape[0]
    class_num=0
    class_name=None
    class_column=np.zeros((N,1))-1
    for i in range(N):
        if i==0:
            class_num=1
            class_name=np.array([class_name,label[0]])
        for j in range(class_num):
            if label[i]==class_name[j]:
                class_column[i]=j
        if class_column[i]==-1:
            class_num=class_num+1
            class_name=np.array([class_name,label[i]])
            class_column[i]=class_num
    Y=np.zeros((N,class_num))
    for i in range(N):
        Y[i,class_column[i]]=1
    return Y,class_name

def li2nsvm_grad(para,X,Y,the_lambda,sigma=[],gamma=[]):
    N=X.shape[0]
    N,D=X.shape
    w=para[0:D]
    b=para[D]
    if len(gamma)==0:
        lambdawgamma=w*the_lambda
    else:
        lambdawgamma=w*(np.array(gamma)+the_lambda)
    Ypred=np.dot(X,w)+b
    active_idx=find(Ypred*(Y<1))
    if len(active_idx)==0:
        dw=2*lambdawgamma
        db=0
        active_E=0
    else:
        active_X=X[active_idx,:]
        active_Y=Y[active_idx,:]
        if len(sigma)==0:
            active_E=Ypred[active_idx]-active_Y
        else:
            active_E=sigma[active_idx,:]*(Ypred[active_idx]-active_Y)
        dw=2*(np.dot(active_E.T,active_X).T)+2*lambdawgamma
        db=2*np.sum(active_E,axis=1)
    df=np.hstack((dw,db),axis=1)
    f=np.dot(active_E.T,active_E)+np.dot(w.T,lambdawgamma)
    return f,df

def lbfgs2(x0, options,  f, sf, X, Y, the_lambda, sigma, gamma):
    history=Params()
    history.obj=[]
    n=x0.shape[0]
    m=options.m
    s=np.zeros((n,m))
    y=np.zeros((n,m))
    alpha=np.zeros(m)
    x_idx=x0
    f_idx,g_idx=li2nsvm_grad(x0,X,Y,the_lambda,sigma,gamma)
    if len(sf)!=0:
        #懒得写了
        pass
    else:
        history.obj.append(f_idx)
    astarguess=options.wolfe.a1
    astar=options.wolfe.a1
    k=-1
    gstar=g_idx
    while True:
        if k<m:
            howmany=k
        else:
            howmany=m
        if howmany<0:
            gamma_k=1
        else:
            gamma_k=(1./rou(howmany))/np.sum(y[:,howmany]*y[:,howmany])
        q=copy.deepcopy(g_idx[:])
        for i in range(howmany,-1,-1):#从howmany倒叙循环到0
            alpha[i]=np.dot(rou[i]*s[:,i].T,q)
            q=q-alpha[i]*y[:,i]
        s_idx=gamma_k*q
        for i in range(howmany):
            beta=np.dot(rou[i]*y[:,i].T,s_idx)
            s_idx+=s[:,i]*(alpha(i)-beta)
        s_idx=s_idx.reshape(x0.shape)
        if k<m:
            s[:,k]=-x_idx[:]
            y[:,k]=-g_idx[:]
        else:
            s[:,:-2]=s[:,1:]
            y[:,:-2]=y[:,1:]
            s[:,-1]=-x_idx[:]
            y[:,-1]=-g_idx[:]
        if k==0:
            astar=options.wolfe.a0
        else:
            if astar<options.wolfe.a1 and abs(astar-options.wolfe.a1)<1e-2*options.wolfe.a1:
                astar=(astar+options.wolfe.a1)/2.
            else:
                astar=(astar+options.wolfe.amax)/2.
        astar,xstar,fstar,gstar=lineSearchWolfe(x_idx, f_idx, g_idx, -s_idx,astar, options.wolfe.amax, options.wolfe.c1,options.wolfe.c2,options.wolfe.maxiter, f,X,Y,the_lambda,sigma,gamma)
        if k<m:
            s[:,k]=xstar[:]+s[:,k]
            y[:,k]=gstar[:]+y[:,k]
            rou[k+1]=1/np.dot(s(:,k+1).T*y(:,k+1))
        else:
            s[:,-1]=xstar[:]+s[:,-1]
            y[:,-1]=gstar[:]+y[:,-1]
            rou[:-2]=rou[1:]
            rou[-1]=1/np.dot(s[:,-1].T,y[:,-1])
        k+=1
        if options.echo==True:
            pass
        if len(sf)!=0:
            #懒得写了
            pass
        else:
            history.obj.append(fstar)
        if abs(gstar[:]).max()*1./(1+abs(fstar))<=options.termination or abs((xstar[:]-x_idx[:])).max()*1./(xstar[:]+2.2204e-16)<=options.xtermination:
            if options.echo==True:
                pass
            retval=0
            xstarfinal=xstar
            if len(sf)==0:
                xstarbest=xstar
            return retval,xstarbest,xstarfinal,history
        if k>=options.maxiter:
            if options.echo==True:
                pass
            retval=1
            xstarfinal=xstar
            if len(sf)==0:
                xstarbest=xstar
            return retval,xstarbest,xstarfinal,history
        f_idx=fstar
        g_idx=gstar
        x_idx=xstar
    return retval,xstarbest,xstarfinal,history

def li2nsvm_lbfgs(X,Y,the_lambda,sigma=[],gamma=[]):
    N,D=X.shape
    w0=np.zeros((D,1))
    b0=0
    wolfe={"a1":0.5,"a0":0.01,"c1":0.0001,"c2":0.9,"maxiter":10,"amax":1.1}
    lbfgs_options={"maxiter":30,"termination":1e-4,"xtermination":1e-4,"m":10,"wolfe":wolfe,"echo":0}
    retval,xstarbest,xstarfinal,history=lbfgs2(np.hstack((w0,b0),axis=1), lbfgs_options, 'li2nsvm_grad', [], X, Y, the_lambda, sigma, gamma)
    w=xstarbest[0:D]
    b=xstarbest[D]
    return w,b

def li2nsvm_multiclass_lbfgs(X,C,the_lambda,gamma=[]):
    Y,class_name=oneofc(C)
    Y=np.sign(Y-0.5)
    dim=X.shape[1]
    cnum=Y.shape[1]
    w=np.zeros((dim,cnum))
    b=np.zeros(cnum)
    for i in range(cnum):
        if len(gamma):
            w[:,i],b[i]=li2nsvm_lbfgs(X,Y[:,i],the_lambda)
        else:
            w[:,i],b[i]=li2nsvm_lbfgs(X,Y[:,i],the_lambda,[],gamma[:,i])
    return w,b,class_name