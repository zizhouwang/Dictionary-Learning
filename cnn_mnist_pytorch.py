'''
基于Pytorch的卷积神经网络MNIST手写数字识别
时间：2020年4月
配置方式：见README.txt
'''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import time
import cv2
from PIL import Image
# 设置plot中文字体
font = {'family': 'MicroSoft YaHei',
        'weight': 'bold',
        'size': '10'}
matplotlib.rc("font", **font)

# 辅助函数-展示图像
def imshow(img,title=None):
    npimg = img.numpy()
    plt.axis("off")
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()

# 设定超参数及常数
learning_rate = 0.0005     #学习率
batch_size = 100           #批处理量
epochs_num = 10            #训练迭代次数
download = True            #数据集加载方式
use_gpu = 0                #CUDA GPU加速  1:使用  0:禁用
is_train = 0               #训练模型  1:重新训练     0:加载现有模型
show_pic = 0               #图像展示  1:展示过程图像  0:关闭图像显示

# 载入MNIST训练集
train_dataset = datasets.MNIST(root='.',                      # 数据集目录
                               train=True,                    # 训练集标记
                               transform=transforms.ToTensor(),  # 转为Tensor变量
                               download=download)

train_loader = DataLoader(dataset=train_dataset,  # 数据集加载
                          shuffle=True,           # 随机打乱数据
                          batch_size=batch_size)  # 批处理量100

# 存入迭代器 展示部分数据
dataiter = iter(train_loader)
batch = next(dataiter)
if show_pic:
    imshow(make_grid(batch[0],nrow=10,padding=2,pad_value=1),'训练集部分数据')

# 初始化卷积神经网络
class MNIST_Network(nn.Module):
    def __init__(self):
        super(MNIST_Network, self).__init__()

        self.conv1 = nn.Conv2d(1,32,kernel_size = 5,padding=2)  # 卷积层
        self.relu1 = nn.ReLU()                                  # 激活函数ReLU
        self.pool1 = nn.MaxPool2d(2,stride=2)                   # 最大池化层

        self.conv2 = nn.Conv2d(32,64,kernel_size = 5,padding=2) # 卷积层
        self.relu2 = nn.ReLU()                                  # 激活函数ReLU
        self.pool2 = nn.MaxPool2d(2,stride=2)                   # 最大池化层

        self.fc3 = nn.Linear(7*7*64,1024)                       # 全连接层
        self.relu3 = nn.ReLU()                                  # 激活函数ReLU

        self.fc4 = nn.Linear(1024,10)                           # 全连接层
        self.softmax4 = nn.Softmax(dim=1)                       # Softmax层

    # 前向传播
    def forward(self, input1):
        # x = self.conv1(input1)
        # x = self.relu1(x)
        # x = self.pool1(x)
        # x = self.conv2(x)
        # x = self.relu2(x)
        # x = self.pool2(x)

        x=input1.reshape(1,-1)
        x=torch.from_numpy(x)
        x=x.to(torch.float32)
        x = self.fc3(x)
        x = self.relu3(x)
        return x.detach().numpy()[0]

        # x = self.fc4(x)
        # x = self.softmax4(x)
        # return x

# 初始化神经网络
net = MNIST_Network()
if use_gpu:           #CUDA GPU加速
    net = net.cuda()
# 加载模型参数
if use_gpu:
    net.load_state_dict(torch.load('.\modelpara.pth'))
else:
    net.load_state_dict(torch.load('.\modelpara.pth', map_location='cpu'))
def get_feature(file_path):
    img = Image.open(file_path)
    img=img.resize((64,49),Image.ANTIALIAS)
    img = np.array(img, dtype=float)[:,:,0]
    feature = net(img)
    return feature