import scipy.io
import pdb
import numpy as np
from sklearn import preprocessing
import time
data = scipy.io.loadmat('clothes5.mat') # 读取mat文件
# print(data.keys())  # 查看mat文件中的所有变量
image_vecs=data['allfeature']
image_vecs=preprocessing.normalize(image_vecs.T, norm='l2').T
labels_mat=data['labels']
labels_index=np.empty((labels_mat.shape[0],labels_mat.shape[1]))
labels_index[:]=-1
for i in range(labels_mat.shape[0]):
    one_label_mat=labels_mat[i]
    one_labels_index=np.where(one_label_mat==1)[0]
    labels_index[i,:one_labels_index.shape[0]]=one_labels_index

t=time.time()

np.random.seed(int(t)%100)
n_classes=labels_index.shape[0]
classes=np.arange(n_classes)
# ind_to_lab_dir={0:"仫佬族",1:"纳西族",2:"怒族",3:"普米族",4:"羌族",5:"撒拉族",6:"畲族"}
lab_to_ind_dir={0:0,1:1,2:2,3:3,4:4}
ind_to_lab_dir={0:0,1:1,2:2,3:3,4:4}
w=54
h=46

start_init_number=30
train_number=120
update_times=90
im_vec_len=w*h
n_atoms = start_init_number
n_neighbor = 8
lamda = 0.5
beta = 1.
gamma = 1.
mu = 2.*gamma
r = 2.
c = 1.

seed = 0 # to save the way initialize dictionary
n_iter_sp = 50 #number max of iteration in sparse coding
n_iter_du = 50 # number max of iteration in dictionary update
n_iter = 15 # number max of general iteration
transform_n_nonzero_coefs=15
n_features = image_vecs.shape[0]

inds_of_file_path_path='inds_of_file_path_wzz_'+str(w)+'_'+str(h)+'_'+str(update_times)+'_'+str(transform_n_nonzero_coefs)+'_'+str(start_init_number)+'.npy'
if os.path.isfile(inds_of_file_path_path):
    inds_of_file_path=np.load(inds_of_file_path_path)
    for i in classes:
        ind_of_lab=lab_to_ind_dir[i]
        labels_of_one_class=inds_of_file_path[ind_of_lab]
        # if i==34 or i==39:    #need to change label rank
        if i==34:    #need to change label rank
            labels_of_one_class.sort()
            # np.random.shuffle(labels_of_one_class)
        if labels_of_one_class.shape[0]<start_init_number:
            print("某个类的样本不足，程序暂停")
            pdb.set_trace()
        inds_of_file_path[ind_of_lab]=labels_of_one_class
else:
    inds_of_file_path=np.empty((n_classes,train_number),dtype=int)
    for i in classes:
        ind_of_lab=lab_to_ind_dir[i]
        labels_of_one_class=labels_index[i][:train_number]
        if labels_of_one_class.shape[0]<start_init_number:
            print("某个类的样本不足，程序暂停")
            pdb.set_trace()
        inds_of_file_path[ind_of_lab]=labels_of_one_class

""" Start the process, initialize dictionary """
Ds=np.empty((n_classes,im_vec_len,n_atoms))
Ws=np.empty((n_classes,n_classes,start_init_number))
As=np.empty((n_classes,n_atoms*n_classes,start_init_number))
Bs=np.empty((n_classes,im_vec_len,start_init_number))
H_Bs=np.empty((n_classes,n_classes,start_init_number))
Q_Bs=np.empty((n_classes,n_atoms*n_classes,start_init_number))
Cs=np.empty((n_classes,start_init_number,start_init_number))
for i in range(n_classes):
    D = image_vecs[:,inds_of_file_path[i][:start_init_number]]
    D = norm_cols_plus_petit_1(D,c)
    Ds[i]=np.copy(D)

print("initializing classifier ... done")
start_t=time.time()
for i in range(update_times):
    for j in range(n_classes):
        j_label=ind_to_lab_dir[j]
        if j==0 and i%10==0:
            print(i)
            sys.stdout.flush()
        D=Ds[j]
        coder = SparseCoder(dictionary=D.T,transform_n_nonzero_coefs=15, transform_algorithm="omp")
        if i==0:
        	Y_init=image_vecs[:,inds_of_file_path[i][:start_init_number]]
            the_H=np.zeros((n_classes,Y_init.shape[1]),dtype=int)
            the_Q=np.zeros((n_atoms*n_classes,Y_init.shape[1]),dtype=int)
            for k in range(Y_init.shape[1]):
                label=y_labelled[k]
                lab_index=lab_to_ind_dir[label]
                the_H[lab_index,k]=1
                the_Q[n_atoms*lab_index:n_atoms*(lab_index+1),k]=1
            X_single =np.eye(D.shape[1]) #X_single的每个列向量是一个图像的稀疏表征
            Bs[j]=np.dot(Y_init[:,start_init_number*j:start_init_number*j+start_init_number],X_single.T)
            H_Bs[j]=np.dot(the_H[:,start_init_number*j:start_init_number*j+start_init_number],X_single.T)
            Q_Bs[j]=np.dot(the_Q[:,start_init_number*j:start_init_number*j+start_init_number],X_single.T)
            Cs[j]=np.linalg.inv(np.dot(X_single,X_single.T))
            Ws[j]=np.dot(H_Bs[j],Cs[j])
            As[j]=np.dot(Q_Bs[j],Cs[j])
        the_B=Bs[j]
        the_H_B=H_Bs[j]
        the_Q_B=Q_Bs[j]
        the_C=Cs[j]
        label_indexs_for_update=np.array(np.where(labels==j_label))[0][:train_number]
        num_of_cla=train_number_of_every_cla[j]
        if num_of_cla>start_init_number+train_number:
            num_of_cla=start_init_number+train_number
        new_index=[label_indexs_for_update[(i+start_init_number)%num_of_cla]]
        im_vec=load_img(file_paths[new_index][0])
        im_vec=im_vec/255.
        new_y=np.array(im_vec,dtype = float)
        new_y=preprocessing.normalize(new_y.T, norm="l2").T*reg_mul
        new_y.reshape(n_features,1)
        new_label=labels[new_index][0]
        new_h=np.zeros((n_classes,1))
        lab_index=lab_to_ind_dir[new_label]
        new_h[lab_index,0]=1
        new_q=np.zeros((n_atoms*n_classes,1))
        new_q[n_atoms*lab_index:n_atoms*(lab_index+1),0]=1
        new_x=(coder.transform(new_y.T)).T
        new_B=the_B+np.dot(new_y,new_x.T)
        new_H_B=the_H_B+np.dot(new_h,new_x.T)
        new_Q_B=the_Q_B+np.dot(new_q,new_x.T)
        new_C=the_C-(np.matrix(the_C)*np.matrix(new_x)*np.matrix(new_x.T)*np.matrix(the_C))/(np.matrix(new_x.T)*np.matrix(the_C)*np.matrix(new_x)+1) #matrix inversion lemma(Woodbury matrix identity)
        Bs[j]=new_B
        H_Bs[j]=new_H_B
        Q_Bs[j]=new_Q_B
        Cs[j]=new_C
        new_D=np.dot(new_B,new_C)
        D=np.copy(new_D)
        Ds[j]=D
        Ws[j]=np.dot(new_H_B,new_C)
        As[j]=np.dot(new_Q_B,new_C)
end_t=time.time()
print("train_time : "+str(end_t-start_t))
D_all=Ds
D_all=D_all.transpose((0,2,1))
D_all=D_all.reshape(-1,im_vec_len).T
np.save("D_all_"+py_file_name+"_mulD_"+str(w)+"_"+str(h)+"_"+str(update_times),D_all)
print("D_all saved")
W_all=Ws
W_all=W_all.transpose((0,2,1))
W_all=W_all.reshape(-1,n_classes).T
np.save("W_all_"+py_file_name+"_mulD_"+str(w)+"_"+str(h)+"_"+str(update_times),W_all)
print("W_all saved")
A_all=As
A_all=A_all.transpose((0,2,1))
A_all=A_all.reshape(-1,n_classes*n_atoms).T
np.save("A_all_"+py_file_name+"_mulD_"+str(w)+"_"+str(h)+"_"+str(update_times),A_all)
print("A_all saved")
pdb.set_trace()