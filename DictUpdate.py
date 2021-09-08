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

class Params:
    pass

def DictUpdate():
    params = struct()
    params.mu = 0
    params.xmu = 0
    params.max_iter = 1
    params.amuco = 0
    params.reg_delta = 1e-20
    params.armijo_a = 0.1
    params.armijo_b = 0.5
    params.armijo_m = 10
    params.armijo_s = 1
    params.scaling = 'diag'
    params.debug = 0
    params.debug_armijo = 0
    params.positive = false
    D = params
    return D