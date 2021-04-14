import numpy as np
from keras.datasets import mnist
import keras
import random
from sklearn.metrics import accuracy_score
from ladder_net import get_ladder_network_fc


# get the dataset
(x_train, y_train), _ = mnist.load_data()
x_train = x_train.reshape(60000, 28*28).astype('float32')/255.0
y_train = np.array(y_train)

tab_unlabelled = np.zeros(5)
tab_test = np.zeros(5)

for random_state in range (5):

    n_classes = 10
    index = list([])
    for i in range (n_classes):
        num_i = 0
        j = 0
        while num_i < 200:
            j = j + 1
            if y_train[j] == i:
                index.append(j)
                num_i = num_i + 1                  
        
    """ random selection for training set (20 labelled samples, 80 unlabelled samples) and testing set (100 samples) """
    index = np.array(index)
    
    random.seed(random_state)
    for i in range (n_classes):
        random.shuffle(index[200*i:200*i + 200])
                
        
    index_l = list([])
    for i in range (n_classes):
        index_l.extend(index[200*i:200*i +20])    
        
        
    y = y_train[index_l]
    x_train_labeled = x_train[index_l]
    
    index_u = list([])
    for i in range (n_classes):
        index_u.extend(index[200*i + 20 :200*i + 100])    
    
    x_train_unlabeled = x_train[index_u]
    
    index_t = list([])
    for i in range (n_classes):
        index_t.extend(index[200*i + 100 :200*i + 200])    
    
    y_test = y_train[index_t]
    x_test = x_train[index_t]
    
        
    y_cate = keras.utils.to_categorical( y )
    y_test_cate = keras.utils.to_categorical( y_test )
    
        
    n_rep =int(x_train_unlabeled.shape[0] / x_train_labeled.shape[0])
    x_train_labeled_rep = np.concatenate([x_train_labeled]*n_rep)
    y_train_labeled_rep = np.concatenate([y_cate]*n_rep)
    
    
    
    # initialize the model 
    inp_size = 28*28 # size of mnist dataset 
    n_classes = 10
      
    model = get_ladder_network_fc(layer_sizes = [ inp_size, 1000 ,500, 250, n_classes ], noise_std = 0.3)
    
    
    # train the model for 100 epochs
    for _ in range(30):
        model.fit([ x_train_labeled_rep , x_train_unlabeled ] , y_train_labeled_rep , epochs=5)
        

    y_test_pr = model.test_model.predict(x_test , batch_size= 1000 )
    tab_test[random_state] = accuracy_score(y_test_cate.argmax(-1) , y_test_pr.argmax(-1))
    
""" Results """ 

print("accuracy :" + str(np.mean(tab_test)))
print("std :" + str(np.std(tab_test)))     
           
