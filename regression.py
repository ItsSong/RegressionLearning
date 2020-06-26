
#---------------简单一元线性回归------梯度下降法

import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("iris_training1.csv",delimiter=",")   #载入数据
x_data = data[1:,0]   #第一列的所有行赋给x_data
y_data = data[1:,3]
plt.scatter(x_data,y_data)   #散点图
plt.show()

lr = 0.0001   #学习率
b = 0         #截距
k = 0         #斜率
epochs = 50   #最大迭代次数

#最小二乘法,计算代价函数
def compute_error(b, k, x_data, y_data):
	totalError = 0
	for i in range(0,len(x_data)):
		totalError += (y_data[i] - (k * x_data[i] + b)) ** 2
	return totalError / float(len(x_data))/2.0   #除以2是为了方便求导

#计算seita0和seita1（截距和斜率）
def gradient_descent_runner(x_data,y_data, b, k, lr, epochs):
	m = float(len(x_data))   #计算总数据量
	for i in range(epochs):
		b_grad = 0
		k_grad = 0
		#计算梯度的总和再求平均
		for j in range(0, len(x_data)):
			b_grad += (1/m) * ((k * x_data[j] + b) - y_data[j])      #对seita0求导
			k_grad += (1/m) * x_data[j] *(((k * x_data[j]) + b) - y_data[j])     #对seita1求导
		#更新b和k
		b = b - (lr * b_grad)
		k = k - (lr * k_grad)
		#每次迭代5次，输出一次图像
		# if i % 5==0:
		# 	print("epochs:",i)
		# 	plt.plot(x_data, y_data, 'b.')
		#     plt.plot(x_data, k*x_data+b, 'r')
		#     plt.show()
	return b, k

print("Starting, b = {0}, k = {1}, error={2}".format(b, k, compute_error(b, k, x_data, y_data)))
print("Running...")
b, k = gradient_descent_runner(x_data, y_data, b, k, lr, epochs)
print("After {0} iterations b = {1}, k={2}, error={3}".format(epochs, b, k, compute_error(b, k, x_data, y_data)))

#画图
plt.plot(x_data, y_data, 'b.')
plt.plot(x_data, k*x_data+b, 'r')
plt.show()