# -*- coding: utf-8 -*-
"""
Created on April 29 2019

@author: Khanh-Hung TRAN
@work : CEA Saclay, France
@email : khanhhung92vt@gmail.com or khanh-hung.tran@cea.fr
"""


import numpy as np
import sys
from numpy.linalg import norm, inv, pinv
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import SparseCoder
from sklearn import preprocessing
import pdb

import warnings
from math import sqrt

from scipy import linalg
from scipy.linalg.lapack import get_lapack_funcs
from joblib import Parallel
from PIL import Image
import cv2

premature = """ Orthogonal matching pursuit ended prematurely due to linear
dependence in the dictionary. The requested precision might not have been met.
"""

def load_img(path):
    im=Image.open(path)    # 读取文件
    # img = cv2.imread(path)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # sift = cv2.xfeatures2d.SIFT_create()
    # pdb.set_trace()
    # _, des = sift.detectAndCompute(gray, None)
    # pdb.set_trace()
    im=np.asarray(im,dtype=float)
    im_dimen=len(im.shape)
    im_vec=None
    if im_dimen==3:
        im_vec=im.transpose((1,0,2)).reshape(-1,1)
    if im_dimen==2:
        im_vec=im.T.reshape(-1,1)
    return im_vec

def remove_zero(Y_one):
    pass
    # Y_one_min=Y_one[Y_one!=0.0].min()
    # Y_one+=Y_one_min
    Y_one+=0.6

def norm_Ys(Y_s):
    for i in range(Y_s.shape[1]):
        remove_zero(Y_s[:,i])
    Y_s = preprocessing.normalize(Y_s.T, norm='l2').T
    return Y_s

def norm_cols(X,eps=np.finfo(float).eps):
	"""
	normalize the columns of a matrix
	"""
	norms = np.sqrt(np.einsum('ij,ij->j', X, X)) + eps
	X /= norms[np.newaxis,:]
	return X

"""	normalize the column of a matrix by unit l2 norm if norm of this column is greater than unit l2 norm """
#从新让X服从 l2正则化 分布
def norm_cols_plus_petit_1(X,the_c):
    eps=np.finfo(float).eps
    norms = norm(X,axis =0) + eps
    
    X_norm = np.copy(X)
    Q = np.where(norms > the_c)[0]
    X_norm[:,Q] = X_norm[:,Q]/norms[Q]*the_c 
    return X_norm

def initialize_D(Y_all, n_atoms, y,n_labelled,seed=0):
    n_classes = len(set(y))
    n_sub_atoms = int(n_atoms/n_classes)
    D = np.zeros((Y_all.shape[0],n_atoms))
    ind_sup = np.arange(n_labelled,Y_all.shape[1])
    np.random.seed(seed)
    np.random.shuffle(ind_sup)

    j = 0
    for i in range (n_classes) :
        ind_i = np.where(y==i)[0]
        np.random.shuffle(ind_i)
        if len(ind_i) >= n_sub_atoms: 
            D[:,i*n_sub_atoms:(i+1)*n_sub_atoms] = np.copy(Y_all[:,ind_i[:n_sub_atoms]])
        else :
            D[:,i*n_sub_atoms:(i+1)*n_sub_atoms] = np.copy(np.hstack((Y_all[:,ind_i],Y_all[:,ind_sup[j:j + n_sub_atoms- len(ind_i)]])))
            j = j + n_sub_atoms- len(ind_i)
    D[:,n_classes*n_sub_atoms:] = np.copy(Y_all[:,ind_sup[j:j + n_atoms - n_classes*n_sub_atoms]])
    return D

def initialize_single_D(Y_all, n_atoms, y,n_labelled,seed=0,D_index=0):
    return np.copy(Y_all[:,n_atoms*D_index:n_atoms*D_index+n_atoms])

def transform(D_all,Y_s,n_nonzero_coefs):
    X_res=np.empty((D_all.shape[1],Y_s.shape[1]))
    for i in range(Y_s.shape[1]):
        X_one=_transform(D_all,Y_s[:,i],n_nonzero_coefs)
        X_res[:,i]=X_one
    return X_res

def _transform(D_all,the_y,n_nonzero_coefs):
    out = gram_omp(
        D_all, the_y, n_nonzero_coefs,
        None, None,
        copy_Gram=False, copy_Xy=False,
        return_path=False)
    x, idx, n_iter = out
    X_one=np.zeros(D_all.shape[1])
    X_one[idx] = x
    return X_one

def DWA_all_init(D_all,W_all,A_all,n_classes,n_atoms,Y_init,y_labelled,lab_to_ind_dir):
    the_H=np.zeros((n_classes,Y_init.shape[1]),dtype=int)
    the_Q=np.zeros((n_atoms*n_classes,Y_init.shape[1]),dtype=int)
    for k in range(Y_init.shape[1]):
        label=y_labelled[k]
        lab_index=lab_to_ind_dir[label]
        the_H[lab_index,k]=1
        the_Q[n_atoms*lab_index:n_atoms*(lab_index+1),k]=1
    X_single=np.zeros((D_all.shape[1],D_all.shape[1]),dtype=float)
    for j in range(D_all.shape[1]):
        X_single[j][j]=1.
    H_Bs=np.dot(the_H,X_single.T)
    Q_Bs=np.dot(the_Q,X_single.T)
    Cs=np.linalg.inv(np.dot(X_single,X_single.T))
    W_all=np.dot(H_Bs,Cs)
    A_all=np.dot(Q_Bs,Cs)
    DWA_all=np.vstack((D_all,W_all,A_all))
    return DWA_all,W_all,A_all,Cs

def train(
    DWA_all,D_all,W_all,A_all,Cs,labels,
    file_paths,inds_of_file_path,
    train_number,start_init_number,update_times,update_index,
    n_classes,n_atoms,n_features,lambda_init,the_lambda,transform_n_nonzero_coefs,
    omp_tag):
    for j in range(n_classes):
        if j==0:
            print(update_index)
            sys.stdout.flush()
        coder = SparseCoder(dictionary=D_all.T,transform_n_nonzero_coefs=transform_n_nonzero_coefs, transform_algorithm='omp')
        label_indexs_for_update=inds_of_file_path[j][:train_number]
        new_index=[label_indexs_for_update[(update_index+start_init_number)%train_number]]
        new_label=labels[new_index][0]
        lab_index=j
        im_vec=load_img(file_paths[new_index][0])
        im_vec=im_vec/255.
        new_y=np.array(im_vec,dtype = float)
        new_y=preprocessing.normalize(new_y.T, norm='l2').T
        new_y=norm_Ys(new_y)
        new_y.reshape(n_features,1)
        new_h=np.zeros((n_classes,1))
        new_h[lab_index,0]=1
        new_q=np.zeros((n_atoms*n_classes,1))
        new_q[n_atoms*lab_index:n_atoms*(lab_index+1),0]=1
        new_yhq=np.vstack((new_y,new_h,new_q))
        new_x=None
        if omp_tag=="true":
            new_x=(coder.transform(new_y.T)).T
        if omp_tag=="wzz":
            new_x=transform(D_all,new_y,transform_n_nonzero_coefs)
        the_C=Cs
        the_u=(1/the_lambda)*np.dot(the_C,new_x)
        gamma=1/(1+np.dot(new_x.T,the_u))
        the_r=new_yhq-np.dot(DWA_all,new_x)
        new_C=(1/the_lambda)*the_C-gamma*np.dot(the_u,the_u.T)
        new_DWA=DWA_all+gamma*np.dot(the_r,the_u.T)
        DWA_all=new_DWA
    part_lambda=(1-update_index/update_times)
    the_lambda=1-(1-lambda_init)*part_lambda*part_lambda*part_lambda
    D_all=DWA_all[0:D_all.shape[0],:]
    W_all=DWA_all[D_all.shape[0]:D_all.shape[0]+W_all.shape[0],:]
    A_all=DWA_all[D_all.shape[0]+W_all.shape[0]:,:]
    D_all=preprocessing.normalize(D_all.T, norm='l2').T
    W_all=preprocessing.normalize(W_all.T, norm='l2').T
    A_all=preprocessing.normalize(A_all.T, norm='l2').T
    DWA_all=np.vstack((D_all,W_all,A_all))
    return DWA_all,D_all,W_all,A_all,the_lambda

def _gram_omp(D_all, the_y, n_nonzero_coefs, tol_0=None, tol=None,
              copy_Gram=True, copy_Xy=True, return_path=False):
    """Orthogonal Matching Pursuit step on a precomputed Gram matrix.

    This function uses the Cholesky decomposition method.

    Parameters
    ----------
    Gram : ndarray of shape (n_features, n_features)
        Gram matrix of the input data matrix.

    Xy : ndarray of shape (n_features,)
        Input targets.

    n_nonzero_coefs : int
        Targeted number of non-zero elements.

    tol_0 : float, default=None
        Squared norm of y, required if tol is not None.

    tol : float, default=None
        Targeted squared error, if not None overrides n_nonzero_coefs.

    copy_Gram : bool, default=True
        Whether the gram matrix must be copied by the algorithm. A false
        value is only helpful if it is already Fortran-ordered, otherwise a
        copy is made anyway.

    copy_Xy : bool, default=True
        Whether the covariance vector Xy must be copied by the algorithm.
        If False, it may be overwritten.

    return_path : bool, default=False
        Whether to return every value of the nonzero coefficients along the
        forward path. Useful for cross-validation.

    Returns
    -------
    gamma : ndarray of shape (n_nonzero_coefs,)
        Non-zero elements of the solution.

    idx : ndarray of shape (n_nonzero_coefs,)
        Indices of the positions of the elements in gamma within the solution
        vector.

    coefs : ndarray of shape (n_features, n_nonzero_coefs)
        The first k values of column k correspond to the coefficient value
        for the active features at that step. The lower left triangle contains
        garbage. Only returned if ``return_path=True``.

    n_active : int
        Number of active features at convergence.
    """
    Gram=np.dot(D_all.T,D_all)
    Xy=np.dot(D_all.T,the_y)
    Gram = Gram.copy('F') if copy_Gram else np.asfortranarray(Gram)

    if copy_Xy or not Xy.flags.writeable:
        Xy = Xy.copy()

    min_float = np.finfo(Gram.dtype).eps
    nrm2, swap = linalg.get_blas_funcs(('nrm2', 'swap'), (Gram,))
    potrs, = get_lapack_funcs(('potrs',), (Gram,))

    indices = np.arange(len(Gram))  # keeping track of swapping
    alpha = Xy
    tol_curr = tol_0
    delta = 0
    gamma = np.empty(0)
    n_active = 0
    residual=np.copy(the_y)

    max_features = len(Gram) if tol is not None else n_nonzero_coefs

    L = np.empty((max_features, max_features), dtype=Gram.dtype)

    L[0, 0] = 1.
    if return_path:
        coefs = np.empty_like(L)

    while True:
        lam = np.argmax(np.abs(alpha))
        if lam < n_active or alpha[lam] ** 2 < min_float:
            # selected same atom twice, or inner product too small
            warnings.warn(premature, RuntimeWarning, stacklevel=3)
            break
        if n_active > 0:
            L[n_active, :n_active] = Gram[lam, :n_active]
            linalg.solve_triangular(L[:n_active, :n_active],
                                    L[n_active, :n_active],
                                    trans=0, lower=1,
                                    overwrite_b=True,
                                    check_finite=False)
            v = nrm2(L[n_active, :n_active]) ** 2
            Lkk = Gram[lam, lam] - v
            if Lkk <= min_float:  # selected atoms are dependent
                warnings.warn(premature, RuntimeWarning, stacklevel=3)
                break
            L[n_active, n_active] = sqrt(Lkk)
        else:
            L[0, 0] = sqrt(Gram[lam, lam])

        Gram[n_active], Gram[lam] = swap(Gram[n_active], Gram[lam])
        Gram.T[n_active], Gram.T[lam] = swap(Gram.T[n_active], Gram.T[lam])
        indices[n_active], indices[lam] = indices[lam], indices[n_active]
        Xy[n_active], Xy[lam] = Xy[lam], Xy[n_active]
        n_active += 1
        # solves LL'x = X'y as a composition of two triangular systems
        gamma, _ = potrs(L[:n_active, :n_active], Xy[:n_active], lower=True,
                         overwrite_b=False)
        residual-=np.dot(D_all[:,indices[:n_active]],gamma)
        print(abs(residual).sum())
        if return_path:
            coefs[:n_active, n_active - 1] = gamma
        beta = np.dot(Gram[:, :n_active], gamma)
        alpha = Xy - beta
        if tol is not None:
            tol_curr += delta
            delta = np.inner(gamma, beta[:n_active])
            tol_curr -= delta
            if abs(tol_curr) <= tol:
                break
        elif n_active == max_features:
            break
    pdb.set_trace()
    if return_path:
        return gamma, indices[:n_active], coefs[:, :n_active], n_active
    else:
        return gamma, indices[:n_active], n_active

def gram_omp(D_all, the_y, n_nonzero_coefs, tol_0=None, tol=None,
              copy_Gram=True, copy_Xy=True, return_path=False):
    """Orthogonal Matching Pursuit step on a precomputed Gram matrix.

    This function uses the Cholesky decomposition method.

    Parameters
    ----------
    Gram : ndarray of shape (n_features, n_features)
        Gram matrix of the input data matrix.

    Xy : ndarray of shape (n_features,)
        Input targets.

    n_nonzero_coefs : int
        Targeted number of non-zero elements.

    tol_0 : float, default=None
        Squared norm of y, required if tol is not None.

    tol : float, default=None
        Targeted squared error, if not None overrides n_nonzero_coefs.

    copy_Gram : bool, default=True
        Whether the gram matrix must be copied by the algorithm. A false
        value is only helpful if it is already Fortran-ordered, otherwise a
        copy is made anyway.

    copy_Xy : bool, default=True
        Whether the covariance vector Xy must be copied by the algorithm.
        If False, it may be overwritten.

    return_path : bool, default=False
        Whether to return every value of the nonzero coefficients along the
        forward path. Useful for cross-validation.

    Returns
    -------
    gamma : ndarray of shape (n_nonzero_coefs,)
        Non-zero elements of the solution.

    idx : ndarray of shape (n_nonzero_coefs,)
        Indices of the positions of the elements in gamma within the solution
        vector.

    coefs : ndarray of shape (n_features, n_nonzero_coefs)
        The first k values of column k correspond to the coefficient value
        for the active features at that step. The lower left triangle contains
        garbage. Only returned if ``return_path=True``.

    n_active : int
        Number of active features at convergence.
    """

    Gram=np.dot(D_all.T,D_all)
    Xy=np.dot(D_all.T,the_y)
    residual=np.copy(the_y)
    resi_reci=1./residual
    resi_reci[resi_reci==np.inf]=0.0
    D_resi=np.dot(D_all.T,resi_reci)
    Gram = Gram.copy('F') if copy_Gram else np.asfortranarray(Gram)
    D_all_T=np.copy(D_all.T)

    if copy_Xy or not Xy.flags.writeable:
        Xy = Xy.copy()

    min_float = np.finfo(Gram.dtype).eps
    nrm2, swap = linalg.get_blas_funcs(('nrm2', 'swap'), (Gram,))
    potrs, = get_lapack_funcs(('potrs',), (Gram,))

    indices = np.arange(len(Gram))  # keeping track of swapping
    alpha = Xy
    tol_curr = tol_0
    delta = 0
    gamma = np.empty(0)
    n_active = 0

    max_features = len(Gram) if tol is not None else n_nonzero_coefs

    L = np.empty((max_features, max_features), dtype=Gram.dtype)

    L[0, 0] = 1.
    idx_used=np.zeros(D_all.shape[1],dtype=int)
    if return_path:
        coefs = np.empty_like(L)

    start_change=15 #for ethnic
    start_change=10 #for clothes origin:62.87% angle:65.21%
    while True:
        # lam = np.argmax(np.abs(Xy))
        lam=None
        if n_active<start_change:
            lam = np.argmin(np.abs(abs(D_resi[n_active:])-the_y.shape[0]))+n_active
        else:
            lam = np.argmax(np.abs(alpha))
        # lam = np.argmin(np.abs(D_resi[n_active:]-the_y.shape[0]))+n_active
        if alpha[lam] ** 2 < min_float:
            # selected same atom twice, or inner product too small
            # warnings.warn(premature, RuntimeWarning, stacklevel=3)
            print("1 found problem")
            sys.stdout.flush()
            break
        if lam < n_active:
            # selected same atom twice, or inner product too small
            # warnings.warn(premature, RuntimeWarning, stacklevel=3)
            print("2 found problem")
            sys.stdout.flush()
            pdb.set_trace()
            break
        if n_active > 0:
            L[n_active, :n_active] = Gram[lam, :n_active]
            linalg.solve_triangular(L[:n_active, :n_active],
                                    L[n_active, :n_active],
                                    trans=0, lower=1,
                                    overwrite_b=True,
                                    check_finite=False)
            v = nrm2(L[n_active, :n_active]) ** 2
            Lkk = Gram[lam, lam] - v
            if Lkk <= min_float:  # selected atoms are dependent
                # warnings.warn(premature, RuntimeWarning, stacklevel=3)
                print("3 found problem")
                sys.stdout.flush()
                break
            L[n_active, n_active] = sqrt(Lkk)
        else:
            L[0, 0] = sqrt(Gram[lam, lam])

        Gram[n_active], Gram[lam] = swap(Gram[n_active], Gram[lam])
        Gram.T[n_active], Gram.T[lam] = swap(Gram.T[n_active], Gram.T[lam])
        indices[n_active], indices[lam] = indices[lam], indices[n_active]
        Xy[n_active], Xy[lam] = Xy[lam], Xy[n_active]
        if n_active<start_change:
            temp=np.copy(D_all_T[lam])
            D_all_T[lam]=D_all_T[n_active]
            D_all_T[n_active]=temp
        n_active += 1
        # solves LL'x = X'y as a composition of two triangular systems
        gamma, _ = potrs(L[:n_active, :n_active], Xy[:n_active], lower=True,
                         overwrite_b=False)
        if n_active<start_change:
            Y_pre=np.dot(D_all_T[:n_active].T,gamma)
            residual=the_y-Y_pre
            resi_temp=np.copy(residual)
            min_temp=abs(resi_temp.min())*2
            resi_temp+=min_temp
            resi_temp = preprocessing.normalize(resi_temp.reshape(1,-1), norm='l2')[0]
            resi_reci=1./resi_temp
            # resi_reci[resi_reci<0]=0.
            resi_reci[resi_reci==np.inf]=0.0
            D_resi=np.dot(D_all.T,resi_reci)
        else:
            beta = np.dot(Gram[:, :n_active], gamma)
            alpha = Xy - beta
        if n_active == max_features:
            break
    if return_path:
        return gamma, indices[:n_active], coefs[:, :n_active], n_active
    else:
        return gamma, indices[:n_active], n_active

def gram_omp_ano(D_all, the_y, n_nonzero_coefs, tol_0=None, tol=None,
              copy_Gram=True, copy_Xy=True, return_path=False):
    Gram=np.dot(D_all.T,D_all)
    Gram_reci=np.linalg.inv(Gram)
    Xy=np.dot(D_all.T,the_y)
    residual=np.copy(the_y)
    resi_reci=1./residual
    resi_reci[resi_reci==np.inf]=0.0
    D_resi=np.dot(D_all.T,resi_reci)
    Gram = Gram.copy('F') if copy_Gram else np.asfortranarray(Gram)

    if copy_Xy or not Xy.flags.writeable:
        Xy = Xy.copy()

    min_float = np.finfo(Gram.dtype).eps
    nrm2, swap = linalg.get_blas_funcs(('nrm2', 'swap'), (Gram,))
    potrs, = get_lapack_funcs(('potrs',), (Gram,))

    indices = np.arange(len(Gram))  # keeping track of swapping
    alpha = Xy
    tol_curr = tol_0
    delta = 0
    gamma = np.empty(0)
    n_active = 0
    idx_used=np.zeros(D_all.shape[1],dtype=int)

    while True:
        # lam = np.argmax(np.abs(Xy))
        lam = np.argmin(np.abs(D_resi-the_y.shape[0]))
        idx_used[lam]=1
        if abs(residual).sum() ** 2 < min_float:
            # selected same atom twice, or inner product too small
            # warnings.warn(premature, RuntimeWarning, stacklevel=3)
            print("1 found problem")
            sys.stdout.flush()
            break

        # indices[n_active], indices[lam] = indices[lam], indices[n_active]
        n_active += 1
        # solves LL'x = X'y as a composition of two triangular systems
        D_part=D_all[:,idx_used==1]
        Dy=np.dot(D_part.T,residual)
        gamma=np.dot(np.linalg.inv(np.dot(D_part.T,D_part)),Dy)
        if n_nonzero_coefs==n_active:
            break
        Y_pre=np.dot(D_part,gamma)
        residual-=Y_pre
        print(abs(residual).sum())
        resi_reci=1./residual
        resi_reci[resi_reci==np.inf]=0.0
        D_resi=np.dot(D_all.T,resi_reci)
        D_resi[idx_used==1]=np.inf
    pdb.set_trace()
    if return_path:
        return gamma, idx_used==1, coefs[:, :n_active], n_active
    else:
        return gamma, idx_used==1, n_active