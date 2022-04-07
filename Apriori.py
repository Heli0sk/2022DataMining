from efficient_apriori import apriori

fp = open('data/aprioridata.csv')
data = fp.readlines()
datalist = []
for line in data:
    line = line.strip()
    line = line.rstrip(',')
    line = line.split(',')
    line = tuple(line)
    datalist.append(line)

itemsets, rules = apriori(datalist, min_support=0.25, min_confidence=0.5)
for itemsetkey in itemsets.keys():
    print("频繁", itemsetkey, "项集有：", itemsets[itemsetkey])
print("============================关联规则============================")
for rule in rules:
    print(rule)
