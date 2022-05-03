from math import log
import operator
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

productDict = {'高': 1, '一般': 2, '低': 3, '中': 2, '帅': 1, '普通': 2, '丑': 3, '胖': 3, '匀称': 2, '瘦': 1, '是': 1, '否': 0}


# 导入数据
def Importdata(datafile):
    dataa = pd.read_excel(datafile)  # datafile是excel文件，所以用read_excel,如果是csv文件则用read_csv
    # 将文本中不可直接使用的文本变量替换成数字

    dataa['income'] = dataa['收入'].map(productDict)  # 将每一列中的数据按照字典规定的转化成数字
    dataa['hight'] = dataa['身高'].map(productDict)
    dataa['look'] = dataa['长相'].map(productDict)
    dataa['shape'] = dataa['体型'].map(productDict)
    dataa['is_meet'] = dataa['是否见面'].map(productDict)

    data = dataa.iloc[:, 5:].values.tolist()  # 取量化后的几列，去掉文本列
    b = dataa.iloc[0:0, 5:-1]
    labels = b.columns.values.tolist()  # 将标题中的值存入列表中

    return data, labels


# 计算数据的熵（entropy）--原始熵
def dataentropy(data, feat):
    lendata = len(data)  # 数据条数
    labelCounts = {}  # 数据中不同类别的条数
    for featVec in data:
        category = featVec[-1]  # 每行数据的最后一个字（叶子节点）
        if category not in labelCounts.keys():
            labelCounts[category] = 0
        labelCounts[category] += 1  # 统计有多少个类以及每个类的数量
    entropy = 0
    for key in labelCounts:
        prob = float(labelCounts[key]) / lendata  # 计算单个类的熵值
        entropy -= prob * log(prob, 2)  # 累加每个类的熵值

    return entropy


# 对数据按某个特征value进行分类
def splitData(data, i, value):
    splitData = []
    for featVec in data:
        if featVec[i] == value:
            rfv = featVec[:i]
            rfv.extend(featVec[i + 1:])
            splitData.append(rfv)

    return splitData


# 选择最优的分类特征
def BestSplit(data):
    numFea = len(data[0]) - 1  # 计算一共有多少个特征，因为最后一列一般是分类结果，所以需要-1
    baseEnt = dataentropy(data, -1)  # 定义初始的熵,用于对比分类后信息增益的变化
    bestGainRate = 0
    bestFeat = -1
    for i in range(numFea):
        featList = [rowdata[i] for rowdata in data]
        uniqueVals = set(featList)
        newEnt = 0
        for value in uniqueVals:
            subData = splitData(data, i, value)  # 获取按照特征value分类后的数据
            prob = len(subData) / float(len(data))
            newEnt += prob * dataentropy(subData, i)  # 按特征分类后计算得到的熵
        info = baseEnt - newEnt  # 原始熵与按特征分类后的熵的差值，即信息增益
        splitonfo = dataentropy(subData, i)  # 分裂信息
        if splitonfo == 0:  # 若特征值相同（eg:长相这一特征的值都是帅），即splitonfo和info均为0，则跳过该特征
            continue
        GainRate = info / splitonfo  # 计算信息增益率
        if (GainRate > bestGainRate):  # 若按某特征划分后，若infoGain大于bestInf，则infoGain对应的特征分类区分样本的能力更强，更具有代表性。
            bestGainRate = GainRate  # 将infoGain赋值给bestInf，如果出现比infoGain更大的信息增益，说明还有更好地特征分类
            bestFeat = i  # 将最大的信息增益对应的特征下标赋给bestFea，返回最佳分类特征
    return bestFeat


def majorityCnt(classList):
    c_count = {}
    for i in classList:
        if i not in c_count.keys():
            c_count[i] = 0
        c_count[i] += 1
    calClassN = sorted(c_count.items(), key=operator.itemgetter(1), reverse=True)  # 按照统计量降序排序

    return calClassN[0][0]  # reverse=True表示降序，因此取[0][0]，即最大值


# 构建树
def createTree(data, labels):
    classList = [rowdata[-1] for rowdata in data]  # 取每一行的最后一列，分类结果（1/0）
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(data[0]) == 1:
        return majorityCnt(classList)
    bestFeat = BestSplit(data)  # 根据信息增益选择最优特征
    bestLab = labels[bestFeat]
    myTree = {bestLab: {}}  # 分类结果以字典形式保存
    del (labels[bestFeat])
    featValues = [rowdata[bestFeat] for rowdata in data]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestLab][value] = createTree(splitData(data, bestFeat, value), subLabels)

    return myTree


# 主程序
datafile = 'data/dateperson/date01.xlsx'  # 文件所在位置
data, labels = Importdata(datafile)  # 导入数据
jc = createTree(data, labels)  # 输出决策树模型结果

print(jc)
