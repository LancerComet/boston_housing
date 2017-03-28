# coding=utf-8

# 使用网格搜索和交叉验证来获取最佳参数.
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# 将要被测试的参数.
params = {
  'name': ('LancerComet', 'Wch'),
  'age': [17, 26]
}

# 所以这里组合有:
# ('LancerComet', 17), ('LancerComet', 26), ('Wch', 17), ('Wch', 26)

# 创建一个 SVC 并进行网格搜索.
svr = SVC()

# 创建分类器.
clf = GridSearchCV(svr, params)

# 交叉验证所有参数.
# 先定义数据集.
data = {
  'name': ('Aniber', 'Lancer'),
  'age': [13, 24]
}

# 验证集.
valid = ()
clf.fit(data, valid)

print clf.best_params_
print clf.best_estimator_
print clf.best_score_

