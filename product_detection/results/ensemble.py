import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


input_1 = pd.read_csv("efficientnet-b1.csv") # 1.4
input_2 = pd.read_csv("resnet152.csv") # 1.1
input_3 = pd.read_csv("resnet152_2.csv") # 1.2
input_4 = pd.read_csv("densenet169.csv") # 1.1
input_5 = pd.read_csv("vgg16_bn.csv") # 1
input_6 = pd.read_csv("resnet101.csv") # 1
input_7 = pd.read_csv("efficientnet-b4.csv") # 1.8
input_8 = pd.read_csv("efficientnet-b3.csv") # 1.6
input_9 = pd.read_csv("efficientnet-b5.csv") # 1.4
input_10 = pd.read_csv("efficientnet-b7.csv")# 1.7


sample = pd.read_csv('efficientnet-b1.csv')

# public predictions
# for i in range(131166):
# private predictions
for i in range(12186):
    
    tmp = [0] * 42
    tmp[int(input_1.iloc[i, 1])] += 1.3
    tmp[int(input_2.iloc[i, 1])] += 1
    tmp[int(input_3.iloc[i, 1])] += 1.1
    tmp[int(input_4.iloc[i, 1])] += 1
    tmp[int(input_5.iloc[i, 1])] += 1
    tmp[int(input_6.iloc[i, 1])] += 1
    tmp[int(input_7.iloc[i, 1])] += 1.3
    tmp[int(input_8.iloc[i, 1])] += 1.3
    tmp[int(input_9.iloc[i, 1])] += 1.3
    tmp[int(input_10.iloc[i, 1])] += 1.3

    max_id = tmp.index(max(tmp))
    if max_id < 10:
        sample.iloc[i, 1] = "0" + str(max_id)
    else:
        sample.iloc[i, 1] = str(max_id)

    if i % 1000 == 0:
        print(i)
sample.to_csv('ensemble.csv', index=False)
