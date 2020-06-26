#!/usr/bin/env python
#-*- coding = utf-8 -*-
#----------------------------标准方程法-----------岭回归

import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

#载入数据
data = genfromtxt(r"LongLey.csv",delimiter=',')

#切分数据
x_data = data[1:,2:]   #第2行、第3列开始的数据
y_data = data[1:,1,np.newaxis]    #第2行、第2列数据
print(x_data)
print("实际值是-----------------")
print(y_data)
print(np.mat(x_data).shape)
print(np.mat(y_data).shape)

#添加偏置项
X_data = np.concatenate((np.ones((16,1)), x_data),axis=1)
#(16,1)是y_data的shape-----16行1列；concatenate是合并矩阵；将16行1列全为1的矩阵和x_data合并
print(X_data.shape)
#print(X_data[:3])   #查看一下前三个

#岭回归标准方程法求解回归参数
def weights(xArr, yArr, lam=0.2):   #lam=0.2是岭系数
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    xTx = xMat.T * xMat    #矩阵乘法
    rxTx = xTx + np.eye(xMat.shape[1]) * lam
    #eye用来生成单位矩阵；xMat.shape[1]是xMat矩阵的列数，这里是7列，因此生成7行7列的单位矩阵
    #计算矩阵的值，如果值为0，说明该矩阵没有逆矩阵
    if np.linalg.det(rxTx) == 0.0:
        print("This matrix cannot do inverse")
        return
    ws = rxTx.I * xMat.T * yMat    #rxTx.I是rxTx的逆矩阵；xMat.T是xMat的转置
    return ws

ws = weights(X_data, y_data)
print(ws)

#计算预测值
print("预测值是---------------------")
print(np.mat(X_data) * np.mat(ws))   #矩阵的乘法



