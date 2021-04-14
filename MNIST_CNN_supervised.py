# -*- coding: utf-8 -*-
"""
Created on April 29 2019
@author: Khanh-Hung TRAN
@work : CEA Saclay, France
@email : khanhhung92vt@gmail.com or khanh-hung.tran@cea.fr
"""

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Activation,GlobalAveragePooling2D
from keras import backend as K
import numpy as np
from scipy.io import loadmat
from sklearn import preprocessing

batch_size = 200
num_classes = 10
epochs = 40

# input image dimensions
img_rows, img_cols = 28, 28

""" Take mnist from keras """  
from keras.datasets import mnist
(data, labels), _ = mnist.load_data()


""" Run 5 times with different random selections """
tab_test = np.zeros(5)
for random_state in range (5):
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
    
    """ random selection for training set (20 labelled samples, 80 unlabelled samples) and testing set (100 samples) """
  
    index = np.array(index)
    np.random.seed(random_state)
    for i in range (n_classes):
        np.random.shuffle(index[200*i:200*i + 200])
                
    k = 20
    index_l = list([])
    for i in range (n_classes):
        index_l.extend(index[200*i:200*i + k])    
    
    y = labels[index_l]
    n_labelled = len(y)
    Y_labelled = np.array(data[index_l],dtype = float)/255.
    
    index_t = list([])
    for i in range (n_classes):
        index_t.extend(index[200*i + 100 :200*i + 200])    
    
    y_test = labels[index_t]
    n_test = len(y_test)
    Y_test = np.array(data[index_t],dtype = float)/255.
    
    
    y_labelled_load = keras.utils.to_categorical(y, n_classes)
    y_test_load = keras.utils.to_categorical(y_test, n_classes)
    
    
    Y_labelled = Y_labelled.reshape(Y_labelled.shape[0], img_rows, img_cols, 1)
    Y_test = Y_test.reshape(Y_test.shape[0], img_rows, img_cols, 1)
    Y_labelled = Y_labelled.astype('float32')
    Y_test = Y_test.astype('float32')
    
       
    input_shape = (img_rows, img_cols, 1)

    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(28,28,1)))
    model.add(Activation('relu'))
    BatchNormalization(axis=-1)
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    BatchNormalization(axis=-1)
    model.add(Conv2D(64,(3, 3)))
    model.add(Activation('relu'))
    BatchNormalization(axis=-1)
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Flatten())
    # Fully connected layer
    
    BatchNormalization()
    model.add(Dense(512))
    model.add(Activation('relu'))
    BatchNormalization()
    model.add(Dropout(0.2))
    model.add(Dense(10))
    
    model.add(Activation('softmax'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])


    model.fit(Y_labelled, y_labelled_load,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1)
    
    score = model.evaluate(Y_test, y_test_load, verbose=0)
    
    tab_test[random_state] = score[1]

""" Results """ 

print("accuracy :" + str(np.mean(tab_test)))
print("std :" + str(np.std(tab_test)))    
