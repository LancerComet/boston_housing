# coding=utf-8

import numpy as np
import pandas as pd
#import visuals as vs
from sklearn.model_selection import ShuffleSplit

# Load the Boston housing dataset
data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)

# TODO: Minimum price of the data
minimum_price = np.min(prices)
print minimum_price

# TODO: Maximum price of the data
maximum_price = np.max(prices)
print maximum_price

# TODO: Mean price of the data
mean_price = np.mean(prices)
print mean_price

# TODO: Median price of the data
median_price = np.median(prices)
print median_price

# TODO: Standard deviation of prices of the data
std_price = np.std(prices)
print std_price

from sklearn.metrics import r2_score

def performance_metric (y_true, y_predict):
    score = r2_score(y_true, y_predict)
    return score

score = performance_metric([3, -0.5, 2, 7, 4.2], [2.5, 0.0, 2.1, 7.8, 5.3])

print 'R2:', score

# 使用 sklearn 的 train_test_split 分割训练集.
# 这是来自 sklearn 官方的一个例子.
from sklearn.model_selection import train_test_split

# 定义 X 与 y.
X = np.arange(10).reshape((5, 2))
y = range(5)
print X
print y

# 分割训练数据与测试数据.
X_train, X_test = train_test_split(
    X, test_size = 0.2
)

print "X_train:", X_train
print "X_test:", X_test

y_train, y_test = train_test_split(y, test_size = 0.5)
print "y_train:", y_train
print "y_test", y_test
