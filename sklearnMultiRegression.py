#!/usr/bin/env python
#-*- coding = utf-8 -*-
#-------------------------------引入sklearn包------多元回归

from sklearn.linear_model import LinearRegression
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#读入数据
data = genfromtxt(r"DrivingAssignment.csv",delimiter=',')
print(data)

#切分数据
x_data = data[1:,:-1]   #第一行是表头，从第二行开始、取每一列的数（前闭后开）。-1代表最后一个列,但不包括最后一个列
y_data = data[1:,-1]   #从第二行开始取，只取最后一列
print(x_data)
print(y_data)

#创建数据
model = LinearRegression()
#拟合模型
model.fit(x_data,y_data)

#查看系数和截距
print("cofficient:",model.coef_)
print("intercept:",model.intercept_)

#测试
x_test = [[102,4]]
predict = model.predict(x_test)
print("predict:",predict)

#画图
ax = plt.figure().add_subplot(111, projection = '3d')
ax.scatter(x_data[:,0], x_data[:,1], y_data, c='r', marker='o', s=100)  #点为红色、三角形、s表示点的大小
x0 = x_data[:,0]
x1 = x_data[:,1]
#生成网络矩阵
x0, x1 = np.meshgrid(x0,x1)
z = model.intercept_ + x0*model.coef_[0] +x1*model.coef_[1]
#画3d图
ax.plot_surface(x0, x1, z)
#设置坐标轴
ax.set_xlabel('Miles')
ax.set_ylabel('Num of Deliveries')
ax.set_zlabel('Time')
plt.show()