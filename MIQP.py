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

m = gp.Model("matrix1")
y=np.random.rand(3)
y=preprocessing.normalize(y.reshape(1,-1), norm='l2')[0]
D_all=np.random.rand(3,100) #字典矩阵D
D_all=preprocessing.normalize(D_all.T, norm='l2').T
obj=QuadExpr()
M=1e+8
T=30
nonzero_num=LinExpr()
y_pres=np.empty(3,dtype=object)
for i in range(3):
    y_pres[i]=QuadExpr()
    y_pres[i]+=-y[i]
for i in range(100):
    z = m.addVar(vtype=GRB.BINARY, name="z" + str(i))
    x = m.addVar(vtype=GRB.CONTINUOUS, name="x" + str(i))
    m.addConstr(-z * M <= x, name="m_x" + str(i) + "1")
    m.addConstr(z * M >= x, name="m_x" + str(i) + "2")
    for j in range(3):
        y_pres[j]+=D_all[j,i]*x
    nonzero_num+=z
for i in range(3):
    v=m.addVar(name="v"+str(i))
    m.addConstr(v<=y_pres[i],name="v_c"+str(i)+"1")
    m.addConstr(v>=y_pres[i],name="v_c"+str(i)+"2")
    obj+=v*v
m.addConstr(nonzero_num<=T,name="t")
m.setObjective(obj, GRB.MINIMIZE)

m.optimize()
z_non_zero_all=0
the_x=np.empty(100)
for v in m.getVars():
    try:
        # if v.varName[0]=="v":
        #     continue
        print('%s' % (v.varName))
        if v.varName[0]=="z":
            z_non_zero_all+=np.sign(abs(v.x))
        if v.varName[0]=="x":
            the_x[int(v.varName[1:])]=v.x
        print('%g' % (v.x))
    except AttributeError:
        print('Encountered an attribute error')
pass