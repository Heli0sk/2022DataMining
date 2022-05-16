from numpy import *
import PIL.Image as image
from sklearn.cluster import KMeans
import numpy as np


def loadData(filePath):
    data = []
    img = image.open(filePath)
    img.show()
    m, n = img.size
    for i in range(m):
        for j in range(n):
            x, y, z = img.getpixel((i, j))
            data.append([x/256.0, y/256.0, z/256.0])

    return mat(data), m, n


def cluster_demo(filePath, savePath, save=False):
    # imgData, row, col = loadData("data/re2.jpg")
    imgData, row, col = loadData(filePath)
    print(row, col)
    print(imgData.shape)
    # print(imgData)

    km = KMeans(n_clusters=5)  # 聚类中心个数为3
    label = km.fit_predict(imgData)
    # 获取簇中心
    centroids = km.cluster_centers_
    print("簇中心: ")
    print(centroids)
    label = label.reshape([row, col])
    print("Labels: ")
    print(label)
    # 创建灰度图保存聚类后的结果
    pic_new = image.new("L", (row, col))

    for i in range(row):
        for j in range(col):
            color = int(256 / (label[i][j] + 1))
            pic_new.putpixel((i, j), color)

    pic_new.show()
    if save:
        pic_new.save(savePath, "JPEG")


if __name__ == "__main__":
    # cluster_demo("data/12week.jpg", "data/km_result")

    data = [[2, 10], [2, 5], [8, 4],
            [5, 8], [7, 5], [6, 4],
            [1, 2], [4, 9]]

    initPoint = np.array(([2, 10], [5, 8], [1, 2]))

    km = KMeans(n_clusters=3, init=initPoint, n_init=1, max_iter=1)
    label = km.fit_predict(data)
    centroids = km.cluster_centers_
    print("簇中心: ")
    print(centroids)
    print(label)
    # print("第一类：A1, C2")
    # print("第二类：A2, C1")
    # print("第三类：A3, B1, B2, B3")
    # print("=" * 17)
    # km2 = KMeans(n_clusters=3)
    # label2 = km2.fit_predict(data)
    # centroids = km2.cluster_centers_
    # print("不指定初始聚类中心时的簇中心: ")
    # print(centroids)

