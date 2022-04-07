
fp = open('data/aprioridata.csv')
data = fp.readlines()
datalist = []
for line in data:
    line = line.strip()
    line = line.rstrip(',')
    line = line.split(',')
    # print(line)
    # line = tuple(line)
    # print(line)
    datalist.append(line)
print(datalist)
