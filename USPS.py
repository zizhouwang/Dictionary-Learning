# -*- coding: utf-8 -*-
"""
Created on April 29 2019

@author: Khanh-Hung TRAN
@work : CEA Saclay, France
@email : khanhhung92vt@gmail.com or khanh-hung.tran@cea.fr
"""



import numpy as np
from SSDL_GU import *
from sklearn import preprocessing
from scipy.io import loadmat
import sys
from sklearn.decomposition import SparseCoder
from sklearn.neighbors import NearestNeighbors


""" Load USPS data and label """
x = loadmat('USPS/USPS.mat') # Need to be reconfigured to your folder
data_sample = x['fea'].T
labels = x['gnd']
labels = labels[:,0] - 1
labels = np.array(labels)

""" Run 6 times with different random selections """
tab_unlabelled = np.zeros(6)
tab_test = np.zeros(6)
for random_state in range (6):

    n_classes = 10
    index = list([])
    for i in range (n_classes):
        num_i = 0
        j = 0
        while num_i < 200:
            j = j + 1
            if labels[j] == i:
                index.append(j)
                num_i = num_i + 1                  
    
    """ random selection for training set (20 labelled samples, 40 unlabelled samples) and testing set (50 samples) """
    index = np.array(index)
    np.random.seed(random_state)
    for i in range (n_classes):
        np.random.shuffle(index[200*i:200*i + 200])
                
        
    index_l = list([])
    for i in range (n_classes):
        index_l.extend(index[200*i:200*i +20])    
    
    y = labels[index_l]
    n_labelled = len(y)
    Y_labelled = np.array(data_sample[:,index_l],dtype = float)
    
    index_u = list([])
    for i in range (n_classes):
        index_u.extend(index[200*i + 20 :200*i + 60])    
    
    y_unlabelled = labels[index_u]
    n_unlabelled = len(y_unlabelled)
    Y_unlabelled = np.array(data_sample[:,index_u],dtype = float)
    
    index_t = list([])
    for i in range (n_classes):
        index_t.extend(index[200*i + 150 :200*i + 200])    
    
    y_test = labels[index_t]
    n_test = len(y_test)
    Y_test = np.array(data_sample[:,index_t],dtype = float)
    
    """ Preprocessing (make each sample have 5 times norm unity) """
    Y_all = np.hstack((Y_labelled,Y_unlabelled))
    Y_all = preprocessing.normalize(Y_all.T, norm='l2').T*5
    Y_test = preprocessing.normalize(Y_test.T, norm='l2').T*5
    
    """ Parameters in optimization  """
    n_atoms = 200
    n_iter = 15
    lamda = 0.3
    beta = 0.5
    gamma = 0.5
    mu = 2.*gamma
    n_neighbor = 8
    seed = 0
    n_iter_du = 50
    r = 1.7
    n_iter_sp = 50
    c = 1.
    n_features = Y_all.shape[0]   
    
    seed = 0 # to save the way initialize dictionary
    n_iter_sp = 50 #number max of iteration in sparse coding
    n_iter_du = 50 # number max of iteration in dictionary update
    n_iter = 15 # number max of general iteration
    
    n_features = Y_all.shape[0]  
    
    """ Contruct matrix LLE """
    Lc = Construct_L_sparse_code(Y_all, n_neighbor)
    
    """ Start the process, initialize dictionary """
   
    D = initialize_D(Y_all, n_atoms, y,n_labelled)
    D = norm_cols_plus_petit_1(D,c)
    
    """ Label matrix for labelled samples """    
    H = -np.ones((n_classes, n_labelled)).astype(float)
    for i in range(n_labelled):
        H[int(y[i]), i] = 1
    
    """ Pseudo-Label matrix for unlabelled samples """
    y_jck = -np.ones((n_classes, n_unlabelled, n_classes)).astype(float)
    for k in range (n_classes):
        y_jck[k,:,k] = 1.
       
    """ Initialize Sparse code X_all """
    coder = SparseCoder(dictionary=D.T,transform_alpha=lamda/2., transform_algorithm='lasso_cd')
    X_all =(coder.transform(Y_all.T)).T
    
    """ Initialize classifier """ 
    W,b = linear_classifier_supervised(H,X_all[:,:n_labelled],mu/gamma)
    
    print("initializing classifier ... done")
    err_total = np.zeros(4*n_iter)
    
    for i in range(n_iter):
        sys.stdout.write("\r" + str(i) + " iterative sur " + str(n_iter))
        sys.stdout.flush()
        
        Score = np.dot(W,X_all[:,n_labelled:]) + b[:,np.newaxis]
        Q_j = y_jck * Score[:,:,np.newaxis] < 1.
        Q_i = H*(np.dot(W,X_all[:,:n_labelled]) + b[:,np.newaxis]) < 1.    
        
        X_unlabelled = np.copy(X_all[:,n_labelled:])
        P = probability_unlabelled_update(y_jck,X_unlabelled,W,b,n_classes,r)
        Pr = P ** r
        err_total[4*i]   =  norm(Y_all - np.dot(D,X_all))**2 + beta*np.trace(X_all.dot(Lc.dot(X_all.T)))  + gamma*classification_score_SP(Q_i,Q_j,y_jck,H,W,b,Pr,X_all[:,:n_labelled],X_all[:,n_labelled:]) + lamda*np.sum(np.abs(X_all)) + mu*(norm(W)**2 + norm(b)**2)
      
        X_all = Beck_Teboulle_proximal_gradient_in_sparse_coding_backpropa(Q_i,Q_j,Y_all, D, X_all, n_labelled, H,y_jck, W,b, Pr, beta,Lc, lamda, gamma, ite_max=n_iter_sp, verbose = False, gap = 0.005)    
        err_total[4*i +1]   =  norm(Y_all - np.dot(D,X_all))**2 + beta*np.trace(X_all.dot(Lc.dot(X_all.T)))  + gamma*classification_score_SP(Q_i,Q_j,y_jck,H,W,b,Pr,X_all[:,:n_labelled],X_all[:,n_labelled:]) + lamda*np.sum(np.abs(X_all)) + mu*(norm(W)**2 + norm(b)**2)
          
        D = optimize_dic_norm_standard_prox(Y_all,D,X_all,c,n_iter_du,False,1e-3)       
        err_total[4*i+2] =  norm(Y_all - np.dot(D,X_all))**2 + beta*np.trace(X_all.dot(Lc.dot(X_all.T))) + gamma*classification_score_SP(Q_i,Q_j,y_jck,H,W,b,Pr,X_all[:,:n_labelled],X_all[:,n_labelled:]) + lamda*np.sum(np.abs(X_all)) + mu*(norm(W)**2 + norm(b)**2)
        
        W,b = multi_binary_classifier_update_original_Wang(Q_i,Q_j,X_all,n_labelled,n_classes,H,y_jck,Pr,mu/gamma)
        err_total[4*i+3] =  norm(Y_all - np.dot(D,X_all))**2 + beta*np.trace(X_all.dot(Lc.dot(X_all.T))) + gamma*classification_score_SP(Q_i,Q_j,y_jck,H,W,b,Pr,X_all[:,:n_labelled],X_all[:,n_labelled:]) + lamda*np.sum(np.abs(X_all)) + mu*(norm(W)**2 + norm(b)**2)
            
   
    """ To see accuracy for labelled samples and for unlabelled samples (in training phase) """
#    X_labelled = X_all[:,:n_labelled]
#    X_unlabelled = X_all[:,n_labelled:]
#    
#    n_correct_train_cl = np.array([y[i] == np.argmax(np.dot(W, X_labelled[:, i])+b)
#                                      for i in range(X_labelled.shape[1])]).nonzero()[0].size/float(X_labelled.shape[1])
#    
#    n_correct_unlabelled_cl = np.array([y_unlabelled[i] == np.argmax(np.dot(W, X_unlabelled[:, i])+b)
#                                      for i in range(X_unlabelled.shape[1])]).nonzero()[0].size/float(X_unlabelled.shape[1])
    
    """ Sparse coding for testing samples if beta != 0 (with manifold structure preservation) """
    n_test = np.shape(Y_test)[1]
    bb1 = np.zeros((n_atoms,n_test))
    neigh = NearestNeighbors(n_neighbors=n_neighbor+1)
    neigh.fit(Y_all.T)
    
    for i in range(n_test):
        a = Y_test[:,i]
        indice_test = neigh.kneighbors(np.atleast_2d(a), n_neighbor, return_distance=False)[0]
        weight_all = np.zeros(Y_all.shape[1])
    
        C = np.zeros((n_neighbor,n_neighbor))
        for k in range (n_neighbor):
            for l in range(k,n_neighbor):
                C[k,l] = np.sum((a - Y_all[:,indice_test[k]]) * (a - Y_all[:,indice_test[l]]))
                if k != l:
                    C[l,k] = C[k,l]
        if np.linalg.cond(C) > 1/sys.float_info.epsilon :
            C = C + (np.trace(C) * 0.1) * np.eye(n_neighbor)
            if np.linalg.cond(C) > 1/sys.float_info.epsilon :
                print("please check matrix C, not invertible")        
            
        invC = np.linalg.linalg.inv(C)
        denum = np.sum(invC)
        for k in range (n_neighbor):            
            weight_all[indice_test[k]] = np.sum(invC[k,:])/denum
        bb1[:,i] = np.dot(X_all,weight_all)
            
        sys.stdout.write("\r" + str(i) + " th sample " + str(n_test) + " done")
        sys.stdout.flush()
    
        
    _Y = np.vstack((Y_test,np.sqrt(beta)*bb1))
    _D = np.vstack((D, np.sqrt(beta) * np.eye(n_atoms)))
    coder = SparseCoder(dictionary=_D.T,transform_alpha=lamda/2., transform_algorithm='lasso_cd')
    X_test =coder.transform(_Y.T).T
    
    """ Classify testing samples by linear learned classifier """
    n_correct_test_cl = np.array([y_test[i] == np.argmax(np.dot(W, X_test[:, i])+b)
                                      for i in range(X_test.shape[1])]).nonzero()[0].size/float(X_test.shape[1]) 
    
    
#    tab_unlabelled[random_state] = n_correct_unlabelled_cl
    tab_test[random_state] = n_correct_test_cl
    
""" Results """ 

print("accuracy :" + str(np.mean(tab_test)))
print("std :" + str(np.std(tab_test)))      
               