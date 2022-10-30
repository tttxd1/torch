import numpy as np
import pandas as pd
import torch
from kmeans_pytorch import kmeans, kmeans_predict
import matplotlib.pyplot as plt

# 导入数据
data = np.random.randint(5,size=(4,6,6))
print(data)

data = torch.from_numpy(data)

cluster_ids_x, cluster_centers = kmeans(
    X=data, num_clusters=1, distance='euclidean')

# 数据集中数据类别所属
print(cluster_ids_x)
# 数据集各类别聚类中心
print(cluster_centers)



