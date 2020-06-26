#!/usr/bin/env python
#-*- coding = utf-8 -*-
#------------------------标准方程法-----------------------一元线性回归
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

#载入数据
data = np.genfromtxt("iris_training1.csv",delimiter=',')
x_data = data[1:,0,np.newaxis]
y_data = data[1:,-1,np.newaxis]
plt.scatter(x_data,y_data)
#plt.show()

print(np.mat(x_data).shape)   #mat表示转变成矩阵数据格式，是单词矩阵的缩写
print(np.mat(y_data).shape)
#给样本添加偏置项/截距
X_data = np.concatenate((np.ones((120,1)), x_data), axis=1)  #ones((120,1)表示120行、1列，且全部是1
#因为"iris_training1.csv"有120行数据，添加偏置项需要匹配
#concatenate是用来合并array的；   axis代表合并的方向
print(X_data.shape)
print(X_data[:3])

#标准方程法求解回归参数
def weight(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    xTx = xMat.T*xMat    #矩阵乘法；   xTx.T是xTx转置
    #计算矩阵的值，如果为0，说明该矩阵没有逆矩阵
    if np.linalg.det(xTx) == 0.0:    #linalg求矩阵的值
        print("This matrix cannot do inverse")
        return
    ws = xTx.I * xMat.T * yMat      #xTx.I是xTx的逆矩阵
    return ws

ws = weight(X_data,y_data)
print(ws)

x_test = np.array([[20],[80]])
y_test = ws[0] + x_test * ws[1]
plt.plot(x_data, y_data, 'b.')
plt.plot(x_test, y_test, 'r')
plt.show()