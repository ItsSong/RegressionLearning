#!/usr/bin/env python
#-*- coding = utf-8 -*-
#-------------------岭回归--------------sklearn
import numpy as np
from numpy import genfromtxt
from sklearn import linear_model
import matplotlib.pyplot as plt

#载入数据
data = genfromtxt(r"LongLey.csv",delimiter=',')
#print(data)

#切分数据
x_data = data[1:,2:]   #第2行、第3列开始的数据
y_data = data[1:,1]    #第2行、第2列数据
# print(x_data)
# print(y_data)

alphas_to_test = np.linspace(0.001,1)    #生成0.001-1之间的数，默认是50个
#创建模型，并保存误差值
model = linear_model.RidgeCV(alphas=alphas_to_test, store_cv_values=True)
#Ridge表示岭回归、CV表示交叉验证
#alphas是岭回归系数
model.fit(x_data, y_data)

#岭系数；会输出最好的那个
print(model.alpha_)
#loss值
print(model.cv_values_.shape)

#画图
#岭系数与loss值的关系图
plt.plot(alphas_to_test, model.cv_values_.mean(axis=0))
#alphas_to_test一共有50个值，即为横坐标；
# 纵坐标是loss值，mean是求平均值，axis=0代表方向
#选取的岭系数值的位置
plt.plot(model.alpha_, min(model.cv_values_.mean(axis=0)), 'ro')
plt.show()

#做预测
print(model.predict(x_data[2,np.newaxis]))   #传入了第3行数据进行预测