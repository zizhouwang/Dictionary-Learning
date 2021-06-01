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
import pdb

import warnings
from math import sqrt

from scipy import linalg
from scipy.linalg.lapack import get_lapack_funcs
from joblib import Parallel

premature = """ Orthogonal matching pursuit ended prematurely due to linear
dependence in the dictionary. The requested precision might not have been met.
"""

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

def transform(D_all,the_y,n_nonzero_coefs):
    out = gram_omp(
        Y_test[:,0], n_nonzero_coefs,
        None, None,
        copy_Gram=False, copy_Xy=False,
        return_path=False)
    x, idx, n_iter = out
    X_one=np.zeros(D_all.shape[1])
    X_one[idx] = x
    return X_one

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
    Y_one_reci=1/the_y
    Yd=np.dot(D_all.T,Y_one_reci)
    Gram = Gram.copy('F') if copy_Gram else np.asfortranarray(Gram)

    if copy_Xy or not Xy.flags.writeable:
        Xy = Xy.copy()

    min_float = np.finfo(Gram.dtype).eps
    nrm2, swap = linalg.get_blas_funcs(('nrm2', 'swap'), (Gram,))
    potrs, = get_lapack_funcs(('potrs',), (Gram,))

    indices = np.arange(len(Gram))  # keeping track of swapping
    alpha = Xy
    residual=np.copy(the_y)
    tol_curr = tol_0
    delta = 0
    gamma = np.empty(0)
    n_active = 0

    max_features = len(Gram) if tol is not None else n_nonzero_coefs

    L = np.empty((max_features, max_features), dtype=Gram.dtype)

    L[0, 0] = 1.
    if return_path:
        coefs = np.empty_like(L)

    while True:
        lam = np.argmax(np.abs(Y_one_reci))
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

    if return_path:
        return gamma, indices[:n_active], coefs[:, :n_active], n_active
    else:
        return gamma, indices[:n_active], n_active

