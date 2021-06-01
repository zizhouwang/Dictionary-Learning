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
import scipy
from sklearn.decomposition import SparseCoder
import pdb

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

""" Construct matrix L by Locally Linear Embedding (LLE) """
def Construct_L_sparse_code(Y_all,n_neighbor):
    
    n_total = Y_all.shape[1]
    """ Locking for neighbours of each sample by calculating all pairwise distances"""
    B = euclidean_distances(Y_all.T, Y_all.T)
    indices = np.argsort(B, axis = 1)
    
    """ Initialization parameter w :
            w[:,i] = min (w[j,i]) ||Y_all[:,i] - sum(w[j,i]Y_all[:,j]) ||^2, j = 1,..., n_neighbor
                s.t sum(w[j,i]) = 1 for each i 
    
    This part is for optimizing parameter w as described in Nonlinear dimensionality
    reduction by locally linear embedding. Sam T. Roweis and Lawrence K. Saul. 2000."""
    
    w = np.zeros((n_total,n_total))
    
    for i in range (n_total):
        C = np.zeros((n_neighbor,n_neighbor))
        for k in range (n_neighbor):
            for l in range(k,n_neighbor):
                C[k,l] = np.sum((Y_all[:,i] - Y_all[:,indices[i,1+k]]) * (Y_all[:,i] - Y_all[:,indices[i,1+l]]))
                if k != l:
                    C[l,k] = C[k,l]
        if np.linalg.cond(C) >= 1/sys.float_info.epsilon :
            C = C + (np.trace(C) * 0.1) * np.eye(n_neighbor)
            if np.linalg.cond(C) >= 1/sys.float_info.epsilon :
                print("C is not invertible, so used pseudo inverse")        
         
        invC = pinv(C)
        denum = np.sum(np.sum(invC))
        for k in range (n_neighbor):            
             w[i,indices[i,k+1]] = np.sum(invC[k,:])/denum
#        sys.stdout.write("\r" + str(i) + " th sample " + str(n_total) + " done")
#        sys.stdout.flush()
    
    """ trace(X_all L X_all.T) = sum( ||X_all[:,i] - sum(w[j,i]*X_all[:,j])||^2), for j = 1,.., n_neighbour
    and i = 1,..,n_total_samples""" 
        
    L = np.eye(n_total) - w - w.T + ((w.T).dot(w)).T   
    return L

""" This function return gradient for labelled samples in sparse coding step """
def grad_for_labelled_in_sparse_coding_fix_Q(Q_i,H,W,b,A_l):
    
    Score = np.dot(W,A_l) + b[:,np.newaxis]    
    return 2 * np.dot(W.T, (Q_i ** 2) * (Score - H))
    
""" This function return gradient for unlabelled samples in sparse coding step """
def grad_for_unlabelled_in_sparse_coding_fix_Q(Q_j,y_jck,Pr,W,b,A_u):
    
    Score = np.dot(W,A_u) + b[:,np.newaxis]
    G = (Q_j ** 2) * (Score[:,:,np.newaxis] - y_jck)
    F = np.zeros((W.shape[1],G.shape[1],G.shape[2]))
    WT = W.T
    for i in range (G.shape[2]):
        F[:,:,i] = 2.* np.dot(WT, G[:,:,i])
    return np.sum(F * Pr[np.newaxis,:,:],axis = 2)

""" This function return classification score while knowing Q_i,Q_j (which are fixed at the beginning of each iteration)"""    
def classification_score_SP(Q_i,Q_j,y_jck,H,W,b,Pr,A_l,A_u):
    return (((((np.dot(W,A_u) + b[:,np.newaxis])[:,:,np.newaxis] - y_jck) * Q_j) ** 2) * Pr[np.newaxis,:,:]).sum() + np.sum((Q_i * (np.dot(W,A_l) + b[:,np.newaxis] - H))**2)       
    
""" This function return classification score with Q_i,Q_j which are first updated """    
def classification_score(y_jck,H,W,b,Pr,A_l,A_u):
    
    Q_i = H*(np.dot(W,A_l) + b[:,np.newaxis]) < 1.
    
    Score_u = np.dot(W,A_u) + b[:,np.newaxis]
    Q_j = y_jck * Score_u[:,:,np.newaxis] < 1.
   
    return (((((np.dot(W,A_u) + b[:,np.newaxis])[:,:,np.newaxis] - y_jck) * Q_j) ** 2) * Pr[np.newaxis,:,:]).sum() + np.sum((Q_i * (np.dot(W,A_l) + b[:,np.newaxis] - H))**2)       
   
""" FISTA with backtracking to optimize Sparse coding problem :
    ||Y_all - D X_all||_F^2 + lamda * ||X_all||_1 + beta *  trace(X_all Lc X_all.T) + gamma *(||Q_i * (W A^l + b[:,np.newaxis] - H)||_F^2 + sum_for_k||Q_j[:,:k] * Pr^(1/2) * (W A^u + b[:,np.newaxis] - y_jck[:,:,k])||_F^2 )  
    """    
def Beck_Teboulle_proximal_gradient_in_sparse_coding_backpropa(Q_i,Q_j,Y_all, D, X_all, n_labelled, H,y_jck, W,b, Pr, beta,Lc, lamda, gamma, ite_max=1000, verbose = False, gap = 0.001):
      
    z = np.zeros_like(X_all)

    if verbose:
        err = np.zeros(ite_max)
        
    x_new = np.copy(X_all)       
    z_new = np.copy(X_all)
    t_new = 1.    
        
    Alpha_D = np.dot(D.T,Y_all)
    Gram_D = np.dot(D.T,D)        
    
    L = 1.
    eta = 1.5
    
    for i in range (ite_max):
        
        z_old = z_new
        t_old = t_new
        x_old = x_new    
        
        gradA = -2. * Alpha_D + 2. * np.dot(Gram_D,z_old) + 2. * beta * np.dot(z_old,Lc)
        gradA[:,:n_labelled] = gradA[:,:n_labelled] + gamma * grad_for_labelled_in_sparse_coding_fix_Q(Q_i,H,W,b,z_old[:,:n_labelled])
        gradA[:,n_labelled:] = gradA[:,n_labelled:] + gamma * grad_for_unlabelled_in_sparse_coding_fix_Q(Q_j,y_jck,Pr,W,b,z_old[:,n_labelled:])
    
        while True:
            
            v =  lamda/L        
            y_n = z_old - (1./L)*gradA
            x_new = np.sign(y_n) * np.maximum(np.abs(y_n) - v,z)
            
            fx_new = norm(Y_all-np.dot(D,x_new))**2 + beta*np.trace(x_new.dot(Lc.dot(x_new.T))) + gamma*classification_score_SP(Q_i,Q_j,y_jck,H,W,b,Pr,x_new[:,:n_labelled],x_new[:,n_labelled:])
            qx_new_z_old = norm(Y_all-np.dot(D,z_old))**2 + beta*np.trace(z_old.dot(Lc.dot(z_old.T))) + gamma*classification_score_SP(Q_i,Q_j,y_jck,H,W,b,Pr,z_old[:,:n_labelled],z_old[:,n_labelled:]) + np.sum((x_new-z_old)*gradA) + L/2.*norm(x_new-z_old)**2  
        
            if fx_new <= qx_new_z_old : 
                break
           
            L = L * eta
        
        t_new = (1 + np.sqrt(4.*(t_old**2.) + 1.))/2.
        lamda_n = 1. + (t_old - 1.)/t_new
        z_new = x_old + lamda_n * (x_new - x_old)
        
        if verbose :
            err[i] = norm(Y_all-np.dot(D,z_new))**2 + beta*np.trace(z_new.dot(Lc.dot(z_new.T))) + gamma*classification_score_SP(Q_i,Q_j,y_jck,H,W,b,Pr,z_new[:,:n_labelled],z_new[:,n_labelled:]) + np.sum((x_new-z_new)*gradA) + L/2.*norm(x_new-z_new)**2 + lamda*np.sum(np.abs(z_new))
             
        if np.max(np.abs(z_old - z_new)) < gap:            
            print("converged after " + str (i) + " iterations")    
            if verbose :
                return z_new,err[0:i]
            else :
                return z_new
            
                
    if i == (ite_max - 1):
        print("number iterative max reached")
          
    if verbose :
        return z_new,err
    else :
        return z_new    

""" FISTA with backtracking to optimize Sparse coding batch set problem :
    ||Y_set - D X_set||_F^2 + lamda * ||X_set||_1 + beta *  trace(X_set Lc_uu X_set.T) + 2 * beta * trace(X_set Lc_ur X_set_com.T) + gamma *(||Q_i_set * (W A_set^l + b[:,np.newaxis] - H_set)||_F^2 + sum_for_k||Q_j_set[:,:k] * Pr_set^(1/2) * (W A_set^u + b[:,np.newaxis] - y_jck_set[:,:,k])||_F^2 )  
    """    
def Beck_Teboulle_proximal_gradient_in_sparse_coding_backpropa_batch(Q_i_set,Q_j_set,Y_set, D, X_set,X_set_com, n_labelled_in_set, H_set,y_jck_set, W,b, Pr_set, beta,Lc_uu,Lc_ur, lamda, gamma,Gram_D,Gram_W, ite_max=1000, verbose = False, gap = 0.001):
    
    z = np.zeros_like(X_set)
    
    if verbose:
        err = np.zeros(ite_max)
        
    x_new = np.copy(X_set)       
    z_new = np.copy(X_set)
    t_new = 1. 
   
    Gram_X_com_Lc = np.dot(X_set_com,Lc_ur.T)
    Alpha_D = np.dot(D.T,Y_set)
  
    L = 0.1
    eta = 1.5
 
    for i in range (ite_max):
        
        z_old = z_new
        t_old = t_new
        x_old = x_new    
        gradA = -2. * Alpha_D + 2. * np.dot(Gram_D,z_old) + 2. * beta * np.dot(z_old,Lc_uu) + 2. * beta* Gram_X_com_Lc
        gradA[:,:n_labelled_in_set] = gradA[:,:n_labelled_in_set] + gamma * grad_for_labelled_in_sparse_coding_fix_Q(Q_i_set,H_set,W,b,z_old[:,:n_labelled_in_set])
        gradA[:,n_labelled_in_set:] = gradA[:,n_labelled_in_set:] + gamma * grad_for_unlabelled_in_sparse_coding_fix_Q(Q_j_set,y_jck_set,Pr_set,W,b,z_old[:,n_labelled_in_set:])
           
        while True:
        
           
            v =  lamda/L       
            y_n = z_old - (1./L)*gradA
            x_new = np.sign(y_n) * np.maximum(np.abs(y_n) - v,z)
        
            fx_new =  norm(Y_set-np.dot(D,x_new))**2 + beta*np.trace(x_new.dot(Lc_uu.dot(x_new.T)) + 2*np.dot(x_new,Lc_ur).dot(X_set_com.T))  + gamma*classification_score_SP(Q_i_set,Q_j_set,y_jck_set,H_set,W,b,Pr_set,x_new[:,:n_labelled_in_set],x_new[:,n_labelled_in_set:])
            qx_new_z_old = norm(Y_set-np.dot(D,z_old))**2 + beta*np.trace(z_old.dot(Lc_uu.dot(z_old.T)) + 2*np.dot(z_old,Lc_ur).dot(X_set_com.T))  + gamma*classification_score_SP(Q_i_set,Q_j_set,y_jck_set,H_set,W,b,Pr_set,z_old[:,:n_labelled_in_set],z_old[:,n_labelled_in_set:]) + np.sum((x_new-z_old)*gradA) + L/2.*norm(x_new-z_old)**2 
        
            if fx_new <= qx_new_z_old : 
                break
           
            L = L * eta
           
        t_new = (1 + np.sqrt(4.*(t_old**2.) + 1.))/2.
        lamda_n = 1. + (t_old - 1.)/t_new
        z_new = x_old + lamda_n * (x_new - x_old)
        
        if verbose :
            err[i] = norm(Y_set-np.dot(D,z_new))**2 + beta*np.trace(z_new.dot(Lc_uu.dot(z_new.T)) + 2*np.dot(z_new,Lc_ur).dot(X_set_com.T)) + gamma*classification_score_SP(Q_i_set,Q_j_set,y_jck_set,H_set,W,b,Pr_set,z_new[:,:n_labelled_in_set],z_new[:,n_labelled_in_set:]) + lamda*np.sum(np.abs(z_new))
             
        if np.max(np.abs(z_old - z_new)) < gap:            
            sys.stdout.write("\r converged after " + str (i) + " iterations")
            sys.stdout.flush()   
            if verbose :
                return z_new,err[0:i]
            else :
                return z_new
            
                
    if i == (ite_max - 1):
        print("number iterative max reached")
          
    if verbose :
        return z_new,err
    else :
        return z_new    

""" return the complement for second parameter in first parameter (universal set) """
def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]

""" FISTA with epoch and batch for sparse coding"""
def Beck_Teboulle_proximal_gradient_in_sparse_coding_by_small_set(Q_i,Q_j,Y_all, D, X_all, n_labelled, H,y_jck, W,b, Pr, beta,Lc, lamda, gamma, n_epochs = 10, n_sets = 10, ite_max_general=1000,verbose = True, gap = 0.001):
    
    if verbose:
        err = np.zeros((n_epochs,n_sets))
    n_total = Y_all.shape[1]
    all_ind = list(np.arange(n_total))
    all_labelled_ind = np.arange(n_labelled)
    all_unlabelled_ind = np.arange(n_labelled,n_total)
    
    Gram_D = np.dot(D.T,D)        
    Gram_W = norm(W) ** 2   
    
    X_all_update = np.copy(X_all)
    
    for j in range (n_epochs):
        np.random.shuffle(all_labelled_ind)
        np.random.shuffle(all_unlabelled_ind)
        
        labelled_in_set = np.array_split(all_labelled_ind,n_sets)
        unlabelled_in_set = np.array_split(all_unlabelled_ind,n_sets)
        
        for i in range (n_sets):
            sys.stdout.write("\r" + "optimizing..." + str(i) + " th of " + str(n_sets) + " in " + str(j) + " th epoch ")
            sys.stdout.flush()
            ind_in_set = np.hstack((labelled_in_set[i],unlabelled_in_set[i]))        
            dif_set_sup_array = np.array(diff(all_ind,list(ind_in_set)))
            n_labelled_in_set = np.shape(labelled_in_set[i])[0]
            
            Y_set = Y_all[:,ind_in_set]
            X_set = X_all_update[:,ind_in_set]
            X_set_com = X_all_update[:,dif_set_sup_array]
            
            Q_i_set = Q_i[:,labelled_in_set[i]]
            Q_j_set = Q_j[:,unlabelled_in_set[i]-n_labelled,:]
            
            H_set = H[:,labelled_in_set[i]]
            y_jck_set =  y_jck[:,unlabelled_in_set[i]-n_labelled,:]
            Pr_set = Pr[unlabelled_in_set[i]-n_labelled,:]
            
            Lc_uu = Lc[ind_in_set[:,None],ind_in_set]
            Lc_ur = Lc[ind_in_set[:,None],dif_set_sup_array]
            
            X_set_up,err_set = Beck_Teboulle_proximal_gradient_in_sparse_coding_backpropa_batch(Q_i_set,Q_j_set,Y_set, D, X_set,X_set_com, n_labelled_in_set, H_set,y_jck_set, W,b, Pr_set, beta,Lc_uu,Lc_ur, lamda, gamma, Gram_D,Gram_W,ite_max_general, verbose = True, gap = gap)
            X_all_update[:,ind_in_set] = np.copy(X_set_up)
            
            if verbose :
                err[j,i] =  norm(Y_all - np.dot(D,X_all_update))**2 + beta*np.trace(X_all_update.dot(Lc.dot(X_all_update.T)))  + gamma*classification_score_SP(Q_i,Q_j,y_jck,H,W,b,Pr,X_all_update[:,:n_labelled],X_all_update[:,n_labelled:]) + lamda*np.sum(np.abs(X_all_update))
  
    
    if verbose :
        return X_all_update,err
    else :
        return X_all_update         

 
""" update probability matrix, according to the work of Xiaobo Wang,
Adaptively unified semisupervised dictionary learning with active points """

def probability_unlabelled_update(y_jck,X_unlabelled,W,b,n_classes,r):
    
    S = np.dot(W,X_unlabelled) + b[:,np.newaxis]
    Q = y_jck * S[:,:,np.newaxis] - 1.
    Q = Q*(Q<=0)
    
    e = np.sum(Q**2,axis = 0)
    
    if r < 1 :
        print("exponent must be equal or greater than 1 ")
    
    if r == 1:
        n_unlabelled = np.shape(X_unlabelled)[1]
        P = np.zeros((n_unlabelled,n_classes))
        t = np.argmax(e, axis = 1)
        for i in range (n_unlabelled):
            P[i,t[i]] = 1.
   
    if r > 1:
        epsilon = sys.float_info.epsilon
        p = (1./(e + epsilon))**(1./(r-1.))
        P = p/p.sum(axis = 1)[:,None]
        
    return P


def multi_binary_classifier_update_original_Wang(Q_i,Q_j,X_all,n_labelled,n_classes,H,y_jck,Pr,mu):
    
    n_atoms = X_all.shape[0]
    n_unlabelled = np.shape(X_all)[1] - n_labelled
      
    A_l_1 = np.vstack((X_all[:,:n_labelled],np.ones((1,n_labelled))))
    A_u_1 = np.vstack((X_all[:,n_labelled:],np.ones((1,n_unlabelled))))
          
    W_next = np.zeros((n_classes,n_atoms+1))    
    
    for c in range (n_classes):
        CP1 = list(np.where(Q_i[c,:] == True)[0])
        CP2 = []
        for j in range (n_unlabelled):
            if (Q_j[c,j,:] == True).all() :
                CP2.append(j)
        
        if len(CP2) > 0 :
           
            p_line_cp2 = np.sum(Pr[CP2,:], axis = 1)
            
            LA = A_l_1[:,CP1].dot(A_l_1[:,CP1].T) + (p_line_cp2 * A_u_1 [:,CP2]).dot(A_u_1 [:,CP2].T)  
            
            
            UN_LA = np.dot(H[c,CP1],A_l_1[:,CP1].T) + np.dot( np.sum(Pr[CP2,:] * y_jck[c,CP2,:],axis = 1), A_u_1[:,CP2].T )
            #UN_LA = np.dot(H[c,CP1],A_l_1[:,CP1].T) + np.sum(np.sum(Pr[CP2,:] * np.array([y_jc -b[c],]*len(CP2)), axis = 1) * A_u_1[:,CP2],axis = 1)
            
            W_next[c,:] = inv(LA+ mu*np.eye(n_atoms+1)).dot(UN_LA)
            
            
        else :
            
            LA =  A_l_1[:,CP1].dot(A_l_1[:,CP1].T) + mu*np.eye(n_atoms) 
              
            UN_LA = np.dot(H[c,CP1],A_l_1[:,CP1].T)
            
            W_next[c,:] = inv(LA+ mu*np.eye(n_atoms+1)).dot(UN_LA)
            
        W_opt = W_next[:,:n_atoms]
        b_opt = W_next[:,n_atoms]    
    
            
            
    return W_opt, b_opt 


"""Here to optimize dictionary by FISTA with backtracking"""
            
def optimize_dic_norm_standard_prox(Y_all,D,X_all,c,ite_max = 2000,verbose = False,gap = 0.001):
    
    """ 
    Using Proximal Method Splitting to optimize the dictionary with constrain,
    each atom of dictionary have length 1:
        ||Y_all - D*X_all||_F^2
        s.t ||D[:,i]|| <= c for all i

    """
    
    if verbose:
        err = np.zeros(ite_max)
        
    x_new = np.copy(D)       
    z_new = np.copy(D)
    t_new = 1.    
        
    YXT = Y_all.dot(X_all.T)   
    XXT = X_all.dot(X_all.T)
    
    L = 1.
    eta = 1.5
         
    for i in range (ite_max):
        
        z_old = z_new
        t_old = t_new
        x_old = x_new    
        
        while True:
            
            gradD =  2. * z_old.dot(XXT) - 2. * YXT  
            
            y_n = z_old - gradD/L
            x_new = norm_cols_plus_petit_1(y_n,c)
            
            fx_new = norm(Y_all-np.dot(x_new,X_all))**2 
            qx_new_z_old = norm(Y_all-np.dot(z_old,X_all))**2 + np.sum((x_new-z_old)*gradD) + L/2.*norm(x_new-z_old)**2  
            
            if fx_new <= qx_new_z_old : 
                break
               
            L = L * eta
        
        t_new = (1 + np.sqrt(4.*(t_old**2.) + 1.))/2.
        lamda_n = 1. + (t_old - 1.)/t_new
        z_new = x_old + lamda_n * (x_new - x_old)
     
        if verbose :
            err[i] = norm(Y_all-np.dot(z_new,X_all))**2
             
        if np.max(np.abs(z_old - z_new)) < gap:            
            print("converged after " + str (i) + " iterations")    
            if verbose :
                return z_new,err[0:i]
            else :
                return z_new
            
    if i == (ite_max - 1):
        print("number iterative max reached")
          
    if verbose :
        return z_new,err
    else :
        return z_new    
          


def return_positive(x):
    l = np.copy(x)
    l[np.where(l<0)] = 0.
    return l 

def dictionary_optimal(XST, gramS, Lagrangian):
    """ return B, where B.T = (S*S.T + Lagrangian)^−1(X*S.T).T 
    SparseCode : S 
    Signal : X
    """

    SSTL = pinv(gramS + np.diag(Lagrangian))
    B = np.dot(SSTL,XST.T).T
    return B

def Dual_Lagrange_function2(XST,gramS,l,c):
    """
    return value of :
        trace( X*S.T*(S*S.T+Lagrangian)^(-1)(X*S.T).T + c*Langrangian )
    """
    return np.trace(np.dot(np.dot(XST,pinv(gramS+np.diag(l))),XST.T)) + c*np.sum(l)


def Gradient_Dic_learing_Lagrange(XST,gramS,L,c):
    """
    return  gradient for Dual Lagrange function2,
    grad = -||X*S.T(S*S.T+Lagrangian)^(-1)ei||^2 + c,
    """
    
    XSTinvSSTL = np.dot(XST,pinv(gramS + np.diag(L)))
    GraD = np.sum(XSTinvSSTL ** 2,axis = 0) - c    
      
    return -GraD    

def Hessien_Dic_learning_Lagrange(XST,gramS,L):

    """
    return hessien for Dual Lagrange function2,
    hess = 2((S*S.T + Lagrangian)^(-1)(X*S.T).T*X*S.T*(S*S.T+Lagrangian)^(-1))i,j((S*S.T+Lagrangian)^(-1))i,j
    """

    invSSTL = pinv(gramS + np.diag(L))
    M = np.dot(XST,invSSTL)
    MMT = np.dot(M.T,M)     
    HessD = -2.*MMT*invSSTL
    return -HessD   

""" Here is method to optimize dictionary by Lagrange multiplier, according to the work of H LEE, in the paper :
    Efficient sparse coding algorithms """
    
""" This function helps to minimize ||X - D S||_F^2 s.t to ||D[:,i]|| <= c for all i """
def dictionary_update_by_Lagrange_dual(X,S,c,ite_max = 200,epsilon = 1e-4,verbose = True):
    
    N = S.shape[0]
    l_old = np.ones(N)
    gam = 0.5
    taux = 0.5
    i = 0
    
    gramS = np.dot(S,S.T)
    XST = np.dot(X,S.T)
    
    if verbose:
        err_du = np.zeros(ite_max)
    
    B_old = Hessien_Dic_learning_Lagrange(XST,gramS,l_old)
    
    H_old = pinv(B_old)
    
    """Calculate new direction"""
    
    Grad = Gradient_Dic_learing_Lagrange(XST,gramS,l_old,c)    
    d = -np.dot(H_old, Grad)
    
    """Backtracking line search"""
    a = 1.
    while Dual_Lagrange_function2(XST,gramS,return_positive(l_old+a*d),c) > Dual_Lagrange_function2(XST,gramS,l_old,c) + a*gam*np.dot(d.T,Grad) :   
        a = taux*a    
    l_new = return_positive(l_old+a*d)

    """Recalculate Hessian or Its approximation"""
    
    B_new = Hessien_Dic_learning_Lagrange(XST,gramS,l_new)
    H_new = pinv(B_new)
        
    """Let be in loop while"""

    while (i<ite_max) and  (np.abs(Dual_Lagrange_function2(XST,gramS,l_new,c) - Dual_Lagrange_function2(XST,gramS,l_old,c))  > epsilon) :
        
        H_old = H_new
        l_old =l_new

        """Calculate new direction"""
    
        Grad = Gradient_Dic_learing_Lagrange(XST,gramS,l_old,c)
        
        d = -np.dot(H_old, Grad)
        
        """Backtracking line search"""
        a = 1.
        while Dual_Lagrange_function2(XST,gramS,return_positive(l_old+a*d),c) > Dual_Lagrange_function2(XST,gramS,l_old,c) + a*gam*np.dot(d.T,Grad) :   
           a = taux*a
        
        l_new = return_positive(l_old+a*d)
        """Recalculate Hessian or Its approximation"""
        B_new = Hessien_Dic_learning_Lagrange(XST,gramS,l_new)
        
        H_new = pinv(B_new)
            
        sys.stdout.write("\r" + "optimizing..." + np.str(i) + "th iteration%")
        sys.stdout.flush()
        
        if verbose :
            B = dictionary_optimal(XST,gramS,l_new)
            err_du[i] = norm(X - np.dot(B,S))**2
        i = i+1
        
    B = dictionary_optimal(XST,gramS,l_new)
    
    if verbose :         
        return B,err_du[:i]
    else :
        return B
    
def dictionary_update_by_Lagrange_dual_with_prox(X,S,c,ite_max = 200,epsilon = 1e-4,verbose = True):
    
    gramS = np.dot(S,S.T)
    XST = np.dot(X,S.T)
    
    N = S.shape[0]
    l_old = np.ones(N)
    
    if verbose:
        err_du = np.zeros(ite_max)
    
    """Calculate new direction"""
      
    x_new = np.copy(l_old)       
    z_new = np.copy(l_old)
    t_new = 1.    
    
    L = 1.
    eta = 1.5
    
    for i in range (ite_max):
        
        z_old = z_new
        t_old = t_new
        x_old = x_new    
        
        while True:
            
            gradD = Gradient_Dic_learing_Lagrange(XST,gramS,l_old,c)      
            
            y_n = z_old - gradD/L
            x_new = return_positive(y_n)
            
            
            fx_new = Dual_Lagrange_function2(XST,gramS,x_new,c)  
            qx_new_z_old = Dual_Lagrange_function2(XST,gramS,z_old,c) + np.sum((x_new-z_old)*gradD) + L/2.*norm(x_new-z_old)**2  
            
            if fx_new <= qx_new_z_old : 
                break
               
            L = L * eta
        
        t_new = (1 + np.sqrt(4.*(t_old**2.) + 1.))/2.
        lamda_n = 1. + (t_old - 1.)/t_new
        z_new = x_old + lamda_n * (x_new - x_old)
     
        if verbose :
            B = dictionary_optimal(XST,gramS,z_new)
            err_du[i] = norm(X - np.dot(B,S))**2
             
        if np.max(np.abs(z_old - z_new)) < epsilon:            
            print("converged after " + str (i) + " iterations")    
            B = dictionary_optimal(XST,gramS,z_new)
            if verbose :    
                return B, err_du[0:i]
            else :
                return B
            
    print("max number iterations ")    
    B = dictionary_optimal(XST,gramS,z_new)
    
    if verbose :         
        return B,err_du[:i]
    else :
        return B
        
"""If the number of atom is over the number of labelled sample, all the labelled samples are employed and the excess is filled up with random selection of unlabelled samples.
   Otherwise, we select randomly samples from the labelled samples  """    
#用样本数据来初始化字典，这里Y_all有1000个样本，使用的是前200个样本，每20个属于一个类别，初始化的字典大小为784*200
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

#Wzz
def initialize_single_D(Y_all, n_atoms, y,n_labelled,seed=0,D_index=0):
    return np.copy(Y_all[:,n_atoms*D_index:n_atoms*D_index+n_atoms])
   
""" This two following top optimize the classifier if the matrix is not invertible  """
def dictionary_leanring_fo(variables,*arg):     
        H = arg[0]
        X = arg[1]
        mu = arg[2]
        variables2d = variables.reshape(H.shape[0],X.shape[0])
        return norm(H - np.dot(variables2d,X))**2 + mu*norm(variables2d)**2

def dictionary_leanring_fprime(variables,*arg):
    H = arg[0]
    X = arg[1]
    mu = arg[2]
    variables2d = variables.reshape(H.shape[0],X.shape[0])
    variable2d_prime =  2*np.dot(np.dot(variables2d,X),X.T) - 2*np.dot(H,X.T) + 2*mu*variables2d
    return variable2d_prime.reshape(-1)    

""" This function optimize ||H - W X_labelled -B||^2 + mu*(||W||^2 + ||b||^2) for W and b """
def linear_classifier_supervised(H,X_labelled,mu):
  
    n_labelled = X_labelled.shape[1]
    n_atoms = X_labelled.shape[0]
    n_classes = H.shape[0]
    
    X1 = np.vstack((X_labelled,np.ones(n_labelled)))
    XXT = np.dot(X1,X1.T) + mu*np.eye(n_atoms+1,n_atoms+1)
    
    if np.linalg.cond(XXT) < 1/sys.float_info.epsilon:
        W1 = np.dot(np.dot(H,X1.T),inv(XXT))
    else :
        print("use BFGS")
        W1 = np.zeros((n_classes,n_atoms+1))
        Wg = scipy.optimize.fmin_l_bfgs_b(dictionary_leanring_fo,fprime = dictionary_leanring_fprime, x0=W1, args=(H,X1,mu), approx_grad=None)
        W1 = Wg[0].reshape(n_classes,n_atoms+1)
        
    
    W = W1[:,:n_atoms]
    b = W1[:,n_atoms]
    return W,b

def sparse_coding_with_LSC(Y_labelled,D,H,W,gamma,lamda):
    _Y = np.vstack((Y_labelled, np.sqrt(gamma) * H))
    _D = np.vstack((D, np.sqrt(gamma) * W))
    
    coder = SparseCoder(dictionary=_D.T,transform_alpha=lamda/2., transform_algorithm='lasso_cd')
    X =(coder.transform(_Y.T)).T
    return X
