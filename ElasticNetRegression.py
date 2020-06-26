#!/usr/bin/env python
#-*- coding = utf-8 -*-
#------------------------弹性网实现回归--------------sklearn

import numpy as np
from numpy import genfromtxt
from sklearn import linear_model

#载入数据
data = genfromtxt(r"LongLey.csv",delimiter=',')

#切分数据
x_data = data[1:,2:]   #第2行、第3列开始的数据
y_data = data[1:,1]    #第2行、第2列数据
print(x_data)
print("实际值是-----------------")
print(y_data)

#创建模型
model = linear_model.ElasticNetCV()   #与LASSO回归类似
#拟合模型
model.fit(x_data, y_data)

#LASSO系数
print(model.alpha_)
#相关系数
print(model.coef_)

#预测
print("预测值-------------------------")
print(model.predict(x_data[-2,np.newaxis]))   #传入倒数第二行的数据，预测其y值