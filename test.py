import numpy as np


def distance(a, b):
    return pow(a[0]-b[0], 2) + pow(a[1]-b[1], 2)


def cal_dis(data, point):
    for item in data:
        print(distance(item, point))


if __name__ == "__main__":

    data = [[2, 10], [2, 5], [8, 4],
            [5, 8], [7, 5], [6, 4],
            [1, 2], [4, 9]]

    cal_dis(data, [1, 2])


