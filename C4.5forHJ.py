from math import log
import operator
import pandas as pd


def LoadData(path):
    data = pd.read_csv(path)
    dataset = data.values.tolist()
    labels = data.columns.values.tolist()
    return dataset, labels


# 统计不同属性数据条目的数量
def calClassN(data):
    num = {}
    for one in data:
        if one[-1] not in num.keys():
            num[one[-1]] = 0
        num[one[-1]] += 1
    return num


# 计算信息熵
def calEntropy(data):
    label_N = calClassN(data)
    numEntries = len(data)
    Entropy = 0.0
    for i in label_N:
        prob = float(label_N[i]) / numEntries
        Entropy -= prob * log(prob, 2)
    return Entropy


# 计算占比最高的类别
def majorClass(data):
    label_N = calClassN(data)
    sortedLabelCount = sorted(label_N.items(), key=operator.itemgetter(1), reverse=True)
    return sortedLabelCount[0][0]


# 取出并返回数据集第i个维度上值为value的子集，i为属性。并且返回的数据去除了第i维
def splitData(data, i, value):
    subDataSet = []
    for one in data:
        if one[i] == value:
            reduceData = one[:i]
            reduceData.extend(one[i + 1:]) # 去除第i维数据
            subDataSet.append(reduceData)
    return subDataSet


# 根据值拆分数据
def splitContinuousDataSet(data, i, value, direction):
    subDataSet = []
    for one in data:
        if direction == 0:
            if one[i] > value:
                reduceData = one[:i]
                reduceData.extend(one[i + 1:])
                subDataSet.append(reduceData)
        if direction == 1:
            if one[i] <= value:
                reduceData = one[:i]
                reduceData.extend(one[i + 1:])
                subDataSet.append(reduceData)
    return subDataSet


# 先从划分属性中找出信息增益高于平均水平的属性，再从中选择增益率最高的
def chooseBestFeat(data, labels):
    baseEntropy = calEntropy(data)  # 数据集整体信息熵
    bestFeat = 0
    baseGainRatio = -1
    numFeats = len(data[0]) - 1  # 特征的个数
    bestSplitDic = {}
    for i in range(numFeats):
        featVals = [example[i] for example in data]
        # 判断属性是否为连续属性，对连续值进行划分选择，否则对离散值进行划分选择
        if type(featVals[0]).__name__ == 'float' or type(featVals[0]).__name__ == 'int':
            sortedFeatVals = sorted(featVals)  # 对选中的属性升序排列
            splitList = []
            for j in range(len(featVals) - 1):
                splitList.append((sortedFeatVals[j] + sortedFeatVals[j + 1]) / 2.0)
            for j in range(len(splitList)):
                newEntropy = 0.0
                gainRatio = 0.0
                splitInfo = 0.0
                value = splitList[j]  # 对二分节点依次进行遍历
                subDataSet0 = splitContinuousDataSet(data, i, value, 0)
                subDataSet1 = splitContinuousDataSet(data, i, value, 1)
                prob0 = float(len(subDataSet0)) / len(data)
                newEntropy -= prob0 * calEntropy(subDataSet0)
                prob1 = float(len(subDataSet1)) / len(data)
                newEntropy -= prob1 * calEntropy(subDataSet1)
                splitInfo -= prob0 * log(prob0, 2)
                splitInfo -= prob1 * log(prob1, 2)
                # 将信息增益转化为增益率
                gainRatio = float(baseEntropy - newEntropy) / splitInfo
                if gainRatio > baseGainRatio:
                    baseGainRatio = gainRatio
                    bestSplit = j
                    bestFeat = i
            bestSplitDic[labels[i]] = splitList[bestSplit]
        else:
            uniqueFeatVals = set(featVals)
            splitInfo = 0.0
            newEntropy = 0.0
            for value in uniqueFeatVals:
                subDataSet = splitData(data, i, value)
                prob = float(len(subDataSet)) / len(data)
                splitInfo -= prob * log(prob, 2)
                newEntropy -= prob * calEntropy(subDataSet)
            if splitInfo == 0:
                gainRatio = 0
            else:
                gainRatio = float(baseEntropy - newEntropy) / splitInfo
            if gainRatio > baseGainRatio:
                bestFeat = i
                baseGainRatio = gainRatio
    if type(data[0][bestFeat]).__name__ == 'float' or type(data[0][bestFeat]).__name__ == 'int':
        bestFeatValue = bestSplitDic[labels[bestFeat]]
    if type(data[0][bestFeat]).__name__ == 'str':
        bestFeatValue = labels[bestFeat]
    return bestFeat, bestFeatValue


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if len(set(classList)) == 1:  # 类别中只剩下一项时停止递归
        return classList[0][0]
    if len(dataSet[0]) == 1:  # 数据集中只剩下一项属性值时生成树
        return majorClass(dataSet)
    bestFeat, bestFeatLabel = chooseBestFeat(dataSet, labels)  # 选择当前数据集中的最优属性
    myTree = {labels[bestFeat]: {}}  # 存放树结构
    subLabels = labels[:bestFeat]
    subLabels.extend(labels[bestFeat + 1:])  # 取出labels中除最优属性外的其他属性
    if type(dataSet[0][bestFeat]).__name__ == 'str':
        featVals = [example[bestFeat] for example in dataSet]
        uniqueVals = set(featVals)
        for value in uniqueVals:
            reduceDataSet = splitData(dataSet, bestFeat, value)
            myTree[labels[bestFeat]][value] = createTree(reduceDataSet, subLabels)
    if type(dataSet[0][bestFeat]).__name__ == 'int' or type(dataSet[0][bestFeat]).__name__ == 'float':
        value = bestFeatLabel
        # 将数据集根据最优属性值进行划分 划分成两个子集
        greaterDataSet = splitContinuousDataSet(dataSet, bestFeat, value, 0)
        smallerDataSet = splitContinuousDataSet(dataSet, bestFeat, value, 1)
        myTree[labels[bestFeat]]['>' + str(value)] = createTree(greaterDataSet, subLabels)
        myTree[labels[bestFeat]]['<=' + str(value)] = createTree(smallerDataSet, subLabels)
    return myTree


if __name__ == '__main__':
    # data, labels = LoadData('data/HJdata.csv')
    fdata = pd.read_csv('data/HJdata.csv').values
    print(fdata.shape)
    features = fdata[:, :3].tolist()
    labels = fdata[:, 3].tolist()
    tree = createTree(features, labels)
    print(tree)
