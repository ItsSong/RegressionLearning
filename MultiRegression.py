
#---------------多元回归---------梯度下降法

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

lr = 0.0001   #学习率
#二元线性回归，有三个参数
theta0 = 0
theta1 = 0
theta2 = 0
epochs = 1000   #最大迭代次数

#最小二乘法计算代价/损失函数
def compute_error(theta0, theta1, theta2, x_data, y_data):
    totalError = 0
    for i in range(0, len(x_data)):
        totalError += (y_data[i] - (theta1 * x_data[i,0] + theta2 * x_data[i,1] + theta0)) ** 2
    return totalError/float(len(x_data))

#求参数
def gradient_descent_runner(x_data, y_data, theta0, theta1, theta2, lr, epochs):
    m = float(len(x_data))  #总数据量
    for i in range(epochs):
        theta0_grad = 0
        theta1_grad = 0
        theta2_grad = 0
        for j in range(0,len(x_data)):   #计算梯度总和，再求平均
            theta0_grad += (1/m) * ((theta1 * x_data[j, 0] + theta2 * x_data[j, 1] + theta0) - y_data[j])
            theta1_grad += (1 / m) * x_data[j,0] * ((theta1 * x_data[j, 0] + theta2 * x_data[j, 1] + theta0) - y_data[j])
            theta2_grad += (1 / m) * x_data[j,1] * ((theta1 * x_data[j, 0] + theta2 * x_data[j, 1] + theta0) - y_data[j])
        theta0 = theta0 - (lr*theta0_grad)
        theta1 = theta1 - (lr * theta1_grad)
        theta2 = theta2 - (lr * theta2_grad)
    return theta0, theta1, theta2

print("Starting theta0 = {0}, theta1 = {1}, theta2 = {2}, error = {3}".format(theta0,theta1,theta2,compute_error(theta0,theta1,theta2,x_data,y_data)))
print("Running...")
theta0, theta1, theta2 = gradient_descent_runner(x_data,y_data,theta0,theta1,theta2,lr,epochs)
print("After {0} iterations theta0 = {1}, theta1={2}, theta2 = {3}, error = {4}".format(epochs, theta0, theta1, theta2, compute_error(theta0,theta1,theta2,x_data,y_data)))

#画图
ax = plt.figure().add_subplot(111, projection = '3d')
ax.scatter(x_data[:,0], x_data[:,1], y_data, c='r', marker='o', s=100)  #点为红色、三角形、s表示点的大小
x0 = x_data[:,0]
x1 = x_data[:,1]
#生成网络矩阵
x0, x1 = np.meshgrid(x0,x1)
z = theta0 + x0*theta1 +x1*theta2
#画3d图
ax.plot_surface(x0, x1, z)
#设置坐标轴
ax.set_xlabel('Miles')
ax.set_ylabel('Num of Deliveries')
ax.set_zlabel('Time')
plt.show()