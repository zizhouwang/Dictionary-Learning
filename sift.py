import cv2
import numpy as np
import numpy.random as rd
from PIL import Image

aa=rd.rand(3,3)
bb=np.dot(aa,(5))

img1 = cv2.imread('data/images/book1.jpeg', 0)

def sift_fea(file_list, cluster_nums, randomState=None):
    features = []
    files = file_list #特征检测
    sift = cv2.xfeatures2d.SIFT_create() #调用SIFT特征提取方法
    for file in files:
        img = Image.open(file).convert('L')
        img=img.resize((40,40),Image.ANTIALIAS)
        img = np.array(img, dtype=np.uint8)
        kp,des = sift.detectAndCompute(img, None) #调用SIFT算法
        # 检测并计算描述符
        # Kp,des=sift.detectAndCompute(gray,None)#检测并计算描述符
        # des =sift.detect(gray, None)# sift.detectAndCompute(gray, None)
        # 找到后可以计算关键点的描述符
        # Kp, des = sift.compute(gray, des)

        reshape_feature = des.reshape(-1, 1)
        features.append(reshape_feature[:,0])

    features = np.array(features) #计算关键点
    return features