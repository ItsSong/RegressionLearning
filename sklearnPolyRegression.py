#!/usr/bin/env python
#-*- coding = utf-8 -*-
#---------------------------------------sklearn包------多项式回归------即曲线

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures   #专门用来生成多项式的
from sklearn.linear_model import LinearRegression

#载入数据
data = np.genfromtxt("salary.csv",delimiter=",")
x_data = data[1:,1]
y_data = data[1:,2]
plt.scatter(x_data,y_data)
#plt.show()

x_data = data[1:,1,np.newaxis]   #改变数据格式，将一维数据转换成二维
y_data = data[1:,2,np.newaxis]
model = LinearRegression()   #创建模型
model.fit(x_data,y_data)     #拟合模型

#画图
plt.plot(x_data,y_data,'b.')  #b代表蓝色，表示实际数据点
plt.plot(x_data,model.predict(x_data), 'r')   #r代表红色，表示预测的线
plt.show()

#定义多项式回归，degree的值可以调节多项式的特征
poly_reg = PolynomialFeatures(degree=5)
#特征处理
x_poly = poly_reg.fit_transform(x_data)
#print(x_poly)      #验证与x_data的不同。可改变degree为不同的1,2,3等等，即可理解特征处理的意思
#定义回归模型
lin_reg = LinearRegression()
#训练/拟合模型
lin_reg.fit(x_poly,y_data)

#画图
plt.plot(x_data,y_data,'b.')
plt.plot(x_data,lin_reg.predict(poly_reg.fit_transform(x_data)),c='r')
plt.title("Truth or Bulff (Polynomial Regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

#画图------测试使用
#点越多，拟合的线越平滑
#z这里随机生成100个点，测试一下与上边10个点拟合的曲线区别
plt.plot(x_data,y_data,'b.')
x_test = np.linspace(1,10,100)   #表示生成100个1-10之间的数
x_test = x_test[:,np.newaxis]    #注意必须转换成二维数据
plt.plot(x_test,lin_reg.predict(poly_reg.fit_transform(x_test)),c='r')
plt.title("Truth or Bulff (Polynomial Regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()


