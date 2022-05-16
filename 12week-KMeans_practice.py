from numpy import *
import numpy as np
from sklearn.cluster import KMeans

data = [[2, 10], [2, 5], [8, 4],
        [5, 8], [7, 5], [6, 4],
        [1, 2], [4, 9]]
label_index = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2"]

# initPoint = np.array(([2, 10], [5, 8], [1, 2]))
# print(initPoint)
class_n = 3
km = KMeans(n_clusters=class_n)  # 聚类中心个数为3
label = km.fit_predict(data)
# 获取簇中心
centroids = km.cluster_centers_
print("簇中心: ")
print(centroids)
print("Labels: ")
print(label)

print("第一类：A2, C1")
print("第二类：A3, B2, B3")
print("第三类：A1, B1, C2")


