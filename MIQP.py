import scipy.io
import time
import pdb
import numpy as np
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
import random
from numpy.matlib import repmat
import copy
import scipy.io as scio
import gurobipy as gp
from gurobipy import GRB
import scipy.sparse as sp
from gurobipy import *

def miqp_ys(the_D,the_Y,T):
    the_X=np.empty((the_D.shape[1],the_Y.shape[1]))
    for i in range(the_Y.shape[1]):
        print("miqp_ys ind: "+str(i))
        the_X[:,i]=miqp(the_D,the_Y[:,i],T)
    return the_X

def miqp(the_D,the_y,T):
    m = gp.Model("miqp")
    the_y=np.random.rand(3)
    the_y=preprocessing.normalize(the_y.reshape(1,-1), norm='l2')[0]
    the_D=np.random.rand(3,100) #字典矩阵D
    the_D=preprocessing.normalize(the_D.T, norm='l2').T
    obj=QuadExpr()
    M=1e+8
    # T=30
    nonzero_num=LinExpr()
    y_pres=np.empty(the_y.shape[0],dtype=object)
    for i in range(the_y.shape[0]):
        y_pres[i]=QuadExpr()
        y_pres[i]+=-the_y[i]
    for i in range(the_D.shape[1]):
        z = m.addVar(vtype=GRB.BINARY, name="z" + str(i))
        x = m.addVar(vtype=GRB.CONTINUOUS, name="x" + str(i))
        m.addConstr(-z * M <= x, name="m_x" + str(i) + "1")
        m.addConstr(z * M >= x, name="m_x" + str(i) + "2")
        for j in range(the_y.shape[0]):
            y_pres[j]+=the_D[j,i]*x
        nonzero_num+=z
    for i in range(the_y.shape[0]):
        v=m.addVar(name="v"+str(i))
        m.addConstr(v<=y_pres[i],name="v_c"+str(i)+"1")
        m.addConstr(v>=y_pres[i],name="v_c"+str(i)+"2")
        obj+=v*v
    m.setObjective(obj, GRB.MINIMIZE)
    m.addConstr(nonzero_num<=T,name="t")

    m.optimize()
    z_non_zero_all=0
    the_x=np.empty(the_D.shape[1])
    for v in m.getVars():
        try:
            # if v.varName[0]=="v":
            #     continue
            if v.varName[0]=="z":
                z_non_zero_all+=np.sign(abs(v.x))
            if v.varName[0]=="x":
                the_x[int(v.varName[1:])]=v.x
        except AttributeError:
            print('Encountered an attribute error')
    err=np.sum(abs(the_y-the_D@the_x))
    return the_x