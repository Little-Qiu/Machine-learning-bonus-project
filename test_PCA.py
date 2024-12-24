from sklearn.svm import SVC                       # svm包中SVC用于分类
from sklearn.decomposition import PCA
import cv2  # opencv库，用于读取图片等操作
import numpy as np
import pandas as pd
import os
from keras.datasets import cifar10
from sklearn.preprocessing import StandardScaler  # 标准差标准化
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split


VSM = pd.read_csv('./VSM.csv', index_col=0)
print(VSM)
print(VSM.shape)
x = VSM.iloc[:, :4613].values
y = VSM[['4614']].values
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.7, test_size=0.3)

T1 = time.time()
# 1、获取数据
# 全部样本
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
print('数据加载完成!')
print('x_train.shape:', x_train.shape)
print('y_train.shape:', y_train.shape)
print('x_test.shape', x_test.shape)
print('y_test.shape', y_test.shape)

# 2、数据标准化
stdScaler = StandardScaler()
x_trainStd = stdScaler.fit_transform(x_train)
x_testStd = stdScaler.fit_transform(x_test)
print('数据标准化完成!')

X = x_trainStd
y = y_train.ravel()
Xt = x_testStd
yt = y_test.ravel()

# 3、PCA降维
pca = PCA()
print("[PCA1] reducing dimensions...")
pca.fit_transform(X)

pca_K = PCA()
print("[PCA2] reducing dimensions...")
pca_K.fit_transform(X)


def Calculate_Best_K_OPT():
    k = 0
    k_opt = 0
    total = sum(pca_K.explained_variance_)
    current_sum = 0
    k_values = []
    var_values = []
    while(k < 500):
        if current_sum/total < 0.99:
            k_opt = k
        current_sum += pca_K.explained_variance_[k]
        k_values.append(k)
        var_values.append(current_sum/total)
        k += 1
    return k, k_opt, k_values, var_values


k, k_opt, k_values, variance_values = Calculate_Best_K_OPT()


plt.figure(figsize=(6, 4))
plt.plot(k_values, variance_values, color='red', linestyle='dashed', marker='o',
         markerfacecolor='orange', markersize=5)
plt.title('Varianza vs. K')
plt.xlabel('K')
plt.axvline(484, 0, 0.95, label='k= ' + str(k_opt) + ' -> Var=' + str(0.99), c='r')
plt.legend()
plt.ylabel('Varianza')

def Calculate_Best_K():
    k = 0
    total = sum(pca.explained_variance_)
    current_sum = 0

    while current_sum / total < 0.99:
        current_sum += pca.explained_variance_[k]
        k += 1
    return k


print(Calculate_Best_K())
plt.show()

T2 = time.time()
print('程序运行时间:%s秒' % (T2 - T1))




