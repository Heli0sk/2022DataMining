import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

'''
|     age    |    income   |  student |   credit_rate  |   buy   |
|----------------------------------------------------------------|
| <=30    0  |  low     0  |  no   0  |  fair       0  | no   0  |
| 31..40  1  |  medium  1  |  yes  1  |  excellent  1  | yes  1  |
| >40     2  |  high    2  |
'''

X = np.array([[0, 2, 0, 0], [0, 2, 0, 1], [1, 2, 0, 0], [2, 1, 0, 0],
              [2, 0, 1, 0], [2, 0, 1, 1], [1, 0, 1, 1], [0, 1, 0, 0],
              [0, 0, 1, 0], [2, 1, 1, 0], [0, 1, 1, 1], [1, 1, 0, 1],
              [1, 2, 1, 0], [2, 1, 0, 1]])
Y = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0])

clf = GaussianNB(priors=None)
# clf.set_params(priors=[0.625, 0.375])
clf.fit(X, Y)
print(clf.predict([[0, 1, 1, 0]]))
print(clf.predict_proba([[0, 1, 1, 0]]))

'''
X = np.array([[-1, -1], [-2, -1], [-3, -2],
              [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])

clf = GaussianNB(priors=None)  # 不输入先验概率
# 获取各个类标记对应的先验概率，返回是列表
print(clf.priors)
# 设置参数，设置各个类标记对应的先验概率
clf.set_params(priors=[0.625, 0.375])
print(clf)
#  获取参数
print(clf.get_params(deep=True))

clf.fit(X, Y)
# 获取先验概率，返回值是数组
print(clf.class_prior_)
# 各个类标签再各个特征上的均值
print(clf.theta_)
# 各个类标签再各个特征上的方差
print(clf.sigma_)
# 各个类标签对应的训练样本数
print(clf.class_count_)

print("=" * 20)

# 直接输出测试集预测的类标记
print(clf.predict([[-0.8, -1], [0, 0], [0, 1]]))
# 输出测试样本再各个类标记预测概率值
print(clf.predict_proba([[-6, -6], [4, 5]]))
# print(clf.predict([[-6, -6], [4, 5]]))

# 输出测试样本在各个类标记上预测概率值对应的对数值
print(clf.predict_log_proba([[-6, -6], [4, 5]]))

score = clf.score([[-6, -6], [-4, -2], [-3, -4], [4, 5]], [1, 1, 2, 2])
print(score)

print(clf.predict([[-6, -6], [-4, -2], [-3, -4], [4, 5]]))
'''
