import numpy as np
import itertools
# from math import mean

with open('/Users/liushihao/Desktop/NMT_Pytorch/raw_data/test.en', 'r') as en:
    res = {}
    short, long = [], []
    i = 0
    for item in en.readlines():
        # print(item)
        item = item.strip().split(' ')
        res[i] = len(item)
        i += 1
    res = sorted(res.items(), key=lambda d: d[1])
    # print(res)
    for i, id in enumerate(res):
        if i < 100:
            short.append(id[0])
        if i > 400:
            long .append(id[0])
    # print(len(long))
    print(short)
en.close()
with open('/Users/liushihao/Desktop/NMT_Pytorch/raw_data/test.en', 'r') as en:

    s_jp = open('/Users/liushihao/Desktop/NMT_Pytorch/test_data/test_short.en', 'w')
    # l_jp = open('/Users/liushihao/Desktop/NMT_Pytorch/test_data/test_long.jp', 'w')
    data = en.readlines()
    for i in short:
        s_jp.write(data[i])
    # for j in long:
    #     l_jp.write(data[j])

# with open('/Users/liushihao/Desktop/NMT_Pytorch/raw_data/test.en', 'r') as en:
#     item = en.readlines()
#     s = open('/Users/liushihao/Desktop/NMT_Pytorch/raw_data/test_half.en', 'w')
#     for i in range(250):
#         s.write(item[i] + '\n')
# j = 0
# s_jp = open('/Users/liushihao/Desktop/NMT_Pytorch/test_data/test_long.en', 'r')
# i = s_jp.readlines()
# j = len(i)
# print(j)

