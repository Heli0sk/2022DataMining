import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])

clf = GaussianNB(priors=None)  # 不输入先验概率
# 获取各个类标记对应的先验概率，返回是列表
print(clf.priors)
# 设置参数，设置各个类标记对应的先验概率
clf.set_params(priors=[0.625, 0.375])
print(clf)
#  获取参数
print(clf.get_params(deep=True))
