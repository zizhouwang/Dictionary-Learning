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

# try:

# Create a new model
m = gp.Model("matrix1")

# Create variables
# x = m.addMVar(shape=3, vtype=GRB.CONTINUOUS, name="x")
# x = m.addMVar(shape=3, vtype=GRB.INTEGER, name="x")

y=np.random.randn(3)
D_all=np.random.randn(3,100) #字典矩阵D
obj=QuadExpr()
M=1e+8
nonzero_num=LinExpr()
T=30
for i in range(100):
# Set objective
    y_pre=QuadExpr()
    for j in range(3):
        x=m.addVar(vtype=GRB.CONTINUOUS, name="x"+str(i)+str(j))
        z=m.addVar(vtype=GRB.BINARY, name="z"+str(i)+str(j))
        m.addConstr(-z*M<=x,name="m_x"+str(i)+str(j)+"1")
        m.addConstr(z*M>=x,name="m_x"+str(i)+str(j)+"2")
        nonzero_num+=z
        y_pre+=y[j]-D_all[j,i]*x
    # d_p = D_all[:,i]
    # y_pre=d_p@x
    v=m.addVar(name="v"+str(i))
    m.addConstr(v<=y_pre,name="v_c"+str(i)+"1")
    m.addConstr(v>=y_pre,name="v_c"+str(i)+"2")
    m.addConstr(nonzero_num<=T,name="t"+str(i))
    obj+=v*v
    m.setObjective(obj, GRB.MINIMIZE)

# Optimize model
m.optimize()

print('Obj: %g' % m.objVal)

# except gp.GurobiError as e:
#     print('Error code ' + str(e.errno) + ": " + str(e))
#
# except AttributeError:
#     print('Encountered an attribute error')