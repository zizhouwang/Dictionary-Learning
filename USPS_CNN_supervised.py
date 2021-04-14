# -*- coding: utf-8 -*-
"""
Created on April 29 2019

@author: Khanh-Hung TRAN
@work : CEA Saclay, France
@email : khanhhung92vt@gmail.com or khanh-hung.tran@cea.fr
"""

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Activation
from scipy.io import loadmat
from sklearn import preprocessing

batch_size = 50
epochs = 40

x = loadmat('USPS/USPS.mat')
data_sample = x['fea'].T
labels = x['gnd']
labels = labels[:,0] - 1
labels = np.array(labels)

""" Run 5 times with different random selections """
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
        index_l.extend(index[200*i:200*i +30])    
    
    y = labels[index_l]
    n_labelled = len(y)
    Y_labelled = np.array(data_sample[:,index_l],dtype = float)
    
    index_u = list([])
    for i in range (n_classes):
        index_u.extend(index[200*i + 30 :200*i + 70])    
    
    y_unlabelled = labels[index_u]
    n_unlabelled = len(y_unlabelled)
    Y_unlabelled = np.array(data_sample[:,index_u],dtype = float)
    
    index_t = list([])
    for i in range (n_classes):
        index_t.extend(index[200*i + 150 :200*i + 200])    
    
    y_test = labels[index_t]
    n_test = len(y_test)
    Y_test = np.array(data_sample[:,index_t],dtype = float)
    
    
    min_max_scaler = preprocessing.MinMaxScaler()
    
    Y_labelled = min_max_scaler.fit_transform(Y_labelled.T).T
    Y_test = min_max_scaler.fit_transform(Y_test.T).T
    
    
    img_rows = 16
    img_cols = 16
    
    
    input_shape = (img_rows, img_cols,1)
    
    x_train = (Y_labelled.T).reshape(Y_labelled.shape[1], img_rows, img_cols, 1)
    x_test = (Y_test.T).reshape(Y_test.shape[1], img_rows, img_cols, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
     
    y = keras.utils.to_categorical(y, n_classes)
    y_test = keras.utils.to_categorical(y_test, n_classes)
    
    
    model = Sequential()
    
    model.add(Conv2D(16, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    BatchNormalization(axis=-1)
    
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Flatten())
    
    BatchNormalization()
    model.add(Dense(128))
    model.add(Activation('tanh'))
    BatchNormalization()
    model.add(Dropout(0.25))
    model.add(Dense(10))
    
    model.add(Activation('softmax'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model.fit(x_train, y,
              batch_size=50,
              epochs=40,
              verbose=1)
    
    
    score = model.evaluate(x_test, y_test, verbose=0)
    
    tab_test[random_state] = score[1]


""" Results """ 

print("accuracy :" + str(np.mean(tab_test)))
print("std :" + str(np.std(tab_test)))

