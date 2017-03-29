# coding=utf-8

# 引入一堆库.
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

# 读取帝都历史房价.
data = pd.read_csv('bj_housing.csv')
features = data.drop('Value', axis = 1)  # 房屋特征.
prices = data['Value']  # 价格.

# 分割数据.
# 测试数据占 25%.
X_train, X_test = train_test_split(features, test_size = 0.25)
y_train, y_test = train_test_split(prices, test_size = 0.25)

# 模型训练函数.
def fitModel (X, y):
  cv_set = ShuffleSplit(n_splits = 10, test_size = 0.25, random_state = 0)
  regressor = DecisionTreeRegressor()
  params = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
  scoring_fnc = make_scorer(r2_score)  # 使用 R2 检查函数创建打分.
  grid = GridSearchCV(regressor, params, scoring = scoring_fnc, cv = cv_set)
  grid = grid.fit(X, y)
  return grid.best_estimator_

# 建立模型.
myModel = fitModel(X_train, y_train)

# 预测一下.
# 100 平方的 3 室 1 厅的 2017 年的 21 层学区破房.
print myModel.predict([100, 3, 1, 1, 2017, 21])
