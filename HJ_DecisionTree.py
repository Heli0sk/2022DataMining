from sklearn import tree
import matplotlib.pyplot as plt


def loadData(path, row, col, bands=3):
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
    '''
    第i行第j列：i * row + j
    return [一通道，二通道，三通道，类别]
    '''

    lim = [[32, 401, 114, 510], [338, 23, 441, 73], [518, 661, 652, 819]]
    for b in range(bands):
        for i in range(32, 114):
            for j in range(402, 511):
                pass
    pass


if __name__ == "__main__":
    org_data = loadData('data/HJ1A-CCD2-450-72-20091015.img', 1440, 813, 3)
    print(org_data)

'''
三类分别是：
陆地：402-511列，33-115行
建筑：24-74列，339-442行
海洋：662-820列，519-653行
'''
