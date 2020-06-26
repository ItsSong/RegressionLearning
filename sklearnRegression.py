
#------------------------------sklearn包------一元线性回归

from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("iris_training1.csv", delimiter=",")
x_data = data[1:,0]   #第一列、从第二行开始所有值赋给x_data___因为第一行是表头
y_data = data[1:,3]
plt.scatter(x_data,y_data)
plt.show()
print(x_data.shape)

x_data = data[1:,0,np.newaxis]  #取出第一列的所有行数据，newaxis转换数据格式
#若不明白newaxis，执行语句print(x_data.shape)，对照x_data前后的格式
y_data = data[1:,3,np.newaxis]

#创建并拟合模型
model = LinearRegression()
model.fit(x_data,y_data)

#画图
plt.plot(x_data,y_data, 'b.')
plt.plot(x_data, model.predict(x_data), 'r')
plt.show()