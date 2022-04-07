import pandas as pd


def Preproces(rPath, wPath, save=False):
    data = pd.read_excel(rPath)
    data = data[['学号', '课程名', '成绩']]
    data.replace('　', 0, inplace=True)
    data = data.sort_values("成绩", ascending=False)
    # 一名同学同一门课程有多个成绩取其最大值
    data.drop_duplicates(subset=["学号", "课程名"], keep="first", inplace=True)
    data = data.pivot(index='学号', columns='课程名', values='成绩')
    data.fillna(0, inplace=True)  # 空值用0填充
    cnt = (data != 0).astype(int).sum(axis=0)
    del_idx = []
    # 删除只有一条非零数据的列
    for i in range(cnt.shape[0]):
        if cnt[i] == 1:
            del_idx.append(i)
    data.drop(data.columns[del_idx], axis=1, inplace=True)
    if save:
        data.to_csv(wPath, index=False)
    return data


def Fluctuation(data, wPath, save=False):
    cols = data.columns
    baseline = []
    for items in cols:
        if items == '学号':
            continue
        baseline.append(data[items][data[items] > 60].mean())
    data = data.values
    # 1为高分, -1为低分
    for i in range(data.shape[0]):
        for j in range(1, data.shape[1]):
            if data[i][j] != 0 and data[i][j] >= baseline[j - 1]:
                data[i][j] = 1
            elif data[i][j] != 0 and data[i][j] < baseline[j - 1]:
                data[i][j] = -1
            elif data[i][j] != 0:
                data[i][j] = 0
    data = pd.DataFrame(data, columns=cols)
    if save:
        data.to_csv(wPath, index=False)
    return data


def Number(data, wPath, save=False):
    data2Encode = data
    NameList = data2Encode.columns  # 每一列的名字
    for i in range(len(NameList)):  # 遍历每一列
        increaingIndex = data2Encode[NameList[i]] > 0  # 成绩高
        data2Encode[NameList[i]][increaingIndex] = float(i * 2)
        decreaingIndex = data2Encode[NameList[i]] < 0  # 成绩低
        data2Encode[NameList[i]][decreaingIndex] = float(i * 2 + 1)
    data2Encode = data2Encode.astype('int')
    data2Encode = data2Encode.T
    if save:
        data2Encode.to_csv(wPath, index=False)
    return data2Encode


# 用于记录非零项
def Screen(orgList):
    res = []
    for i in orgList:
        if i != 0:
            res.append(i)
    return res


def loadData(path):
    # data = pd.read_excel(path)
    data = pd.read_csv(path)
    input = []
    for i in data.columns:  # 处理每一列的数据
        dataRow = list(data[i])  # 获得一列的数据
        dataRow = Screen(dataRow)  # 记录下非零项
        input.append(dataRow)
    return input


# 构建候选1项集C1，参数为原始事务集
def createC1(dataSet):
    C1 = set()
    for i in dataSet:
        for item in i:
            C1.add(frozenset([item]))
    return C1


# 通过候选项集Ck得到频繁项集Lk
# 参数依次为：原数据集，候选集Ck 最小支持度
def createLk(data, Ck, minSupport):
    count = {}  # 候选集计数
    for tid in data:
        # print('tid=',tid)
        for can in Ck:
            # 对于每一个候选项集can，检查是否是transaction的一部分
            # 即该候选can是否得到transaction的支持
            if can.issubset(tid):
                if can not in count.keys():
                    count[can] = 1
                else:
                    count[can] += 1
    N = float(len(data))
    Lk = []  # 候选集项Cn生成的频繁项集Lk
    supportData = {}  # 候选集项Cn的支持度字典
    # 计算候选项集的支持度, supportData key:候选项， value:支持度
    for key in count:  # 每个项集的支持度
        support = count[key] / N
        # 满足最小支持度的项集加入LK
        if support >= minSupport:
            Lk.append(key)
        supportData[key] = support
    return Lk, supportData


# 连接操作，将频繁Lk-1项集通过拼接转换为候选k项集
def aprioriGen(Lk_1, k):
    Ck = []
    lenLk = len(Lk_1)
    for i in range(lenLk):
        L1 = list(Lk_1[i])[:k - 2]
        L1.sort()
        for j in range(i + 1, lenLk):
            # 前k-2个项相同时，将两个集合合并
            L2 = list(Lk_1[j])[:k - 2]
            L2.sort()
            if L1 == L2:
                Ck.append(Lk_1[i] | Lk_1[j])
    return Ck


# 返回所有满足条件的频繁项集的列表，和所有候选项集的支持度信息
def apriori(dataSet, minSupport=0.5):
    C1 = createC1(dataSet)  # 初始候选集C1
    # 构建初始的频繁项集
    L1, supportData = createLk(dataSet, C1, minSupport)
    L = [L1]
    k = 2  # 最初的L1中的每个项集含有一个元素，新生成的项集应该含有2个元素，所以 k=2
    while (len(L[k - 2]) > 0):
        Lk_1 = L[k - 2]
        Ck = aprioriGen(Lk_1, k)
        Lk, supK = createLk(dataSet, Ck, minSupport)
        supportData.update(supK)  # 将新的项集的支持度数据加入原来的总支持度字典中
        L.append(Lk)  # 将符合最小支持度要求的项集加入L
        k += 1  # 下一次循环生成k+1项集

    return L, supportData


# 生成候选规则集合：计算规则的可信度以及找到满足最小可信度要求的规则
def getinfo(num):
    data = pd.read_csv('data/score3.csv')
    NameList = data.columns  # 每一列的名字
    good = NameList[int(num / 2)]
    up = num % 2
    ans = good
    if up:
        ans = ans + "  分数低"
    else:
        ans = ans + "  分数高"
    return ans


# 针对项集中只有两个元素时，计算可信度
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = []  # 返回一个满足最小可信度要求的规则列表
    for conseq in H:  # 后件，遍历 H中的所有项集并计算它们的可信度值
        conf = supportData[freqSet] / supportData[freqSet - conseq]  # 可信度计算，结合支持度数据
        if conf >= minConf:
            cause = list(freqSet - conseq)
            result = list(conseq)
            print(getinfo(cause[0]), "--->>>", getinfo(result[0]))
            # 如果某条规则满足最小可信度值,那么将这些规则输出
            brl.append((freqSet - conseq, conseq, conf))  # 添加到规则里，brl 是前面通过检查的 bigRuleList
            prunedH.append(conseq)  # 同样需要放入列表到后面检查
    return prunedH


# 合并
# 参数:一个是频繁项集,另一个是可以出现在规则右部的元素列表 H
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if (len(freqSet) > (m + 1)):  # 频繁项集元素数目大于单个集合的元素数
        Hmp1 = aprioriGen(H, m + 1)  # 存在不同顺序、元素相同的集合，合并具有相同部分的集合
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)  # 计算可信度
        if (len(Hmp1) > 1):
            # 满足最小可信度要求的规则列表多于1,则递归来判断是否可以进一步组合这些规则
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)


# 参数：频繁项集列表、包含那些频繁项集支持数据的字典、最小可信度阈值
def generateRules(L, supportData, minConf=0.7):
    bigRuleList = []  # 存储所有的关联规则
    for i in range(1, len(L)):  # 只获取有两个或者更多集合的项目，从1,即第二个元素开始，L[0]是单个元素的
        # 两个及以上的才可能有关联一说，单个元素的项集不存在关联问题
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            # 该函数遍历L中的每一个频繁项集并对每个频繁项集创建只包含单个元素集合的列表H1
            if (i > 1):
                # 如果频繁项集元素数目超过2,那么会考虑对它做进一步的合并
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:  # 第一层时，后件数为1
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)  # 调用函数2
    return bigRuleList


if __name__ == "__main__":
    org_data = Preproces("data/成绩表.xls", "data/score2.csv", save=True)
    data = Fluctuation(org_data, 'data/score3.csv', save=True)
    data = Number(data, 'data/score4.csv', save=True)
    dataset = loadData('data/score4.csv')
    # 用apriori算法得到关联规则
    L, supportData = apriori(dataset, minSupport=0.45)
    print("----------关联规则-----------")
    rules = generateRules(L, supportData, minConf=0.7)
