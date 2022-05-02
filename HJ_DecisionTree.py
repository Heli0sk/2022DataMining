from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd


def loadData(path, row, col, bands=3):
    """
    :param path:  文件路径
    :param row:   行数
    :param col:   列数
    :param bands: 通道数
    :return: list [[通道1], [通道2], [通道3]]
    """
    data = open(path, 'rb')
    bd = []
    for i in range(bands):
        bd.append([])
    for i in range(bands):
        for j in range(row*col):
            tmp = int.from_bytes(data.read(1), byteorder='big', signed=False)
            bd[i].append(tmp)
    data.close()
    return bd


def preProcess(data, row, col, bands=3):
    """
    :param data:
    :param row:
    :param col:
    :param bands: 通道数
    :return:      [一通道，二通道，三通道，类别]
    """
    # 第i行第j列：i * row + j

    lim = [[32, 401, 114, 510], [338, 23, 441, 73], [518, 661, 652, 819]]
    band = [[], [], []]
    clas = []

    for b in range(bands):
        for i in range(32, 115):
            for j in range(401, 511):
                band[b].append(data[b][i*row+j])
                clas.append(0)
        for i in range(338, 442):
            for j in range(23, 74):
                band[b].append(data[b][i*row+j])
                clas.append(1)
        for i in range(518, 653):
            for j in range(662, 820):
                band[b].append(data[b][i*row+j])
                clas.append(2)
    # return band, clas
    # band[0], band[1], band[2], class
    res = pd.DataFrame(list(zip(band[0], band[1], band[2], clas)), columns=['band1', 'band2', 'band3', 'class'])
    res.to_csv('data/HJdata.csv', index=False)
    return res


if __name__ == "__main__":
    org_data = loadData('data/HJ1A-CCD2-450-72-20091015.img', 1440, 813, 3)
    # print(org_data)
    data = preProcess(org_data, 1440, 813, 3)
    print(data.head())

'''
三类分别是：
陆地：402-511列，33-115行
建筑：24-74列，339-442行
海洋：662-820列，519-653行
'''
