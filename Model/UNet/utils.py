import numpy as np
from sklearn.cluster import KMeans


class Memory():
    def __init__(self,represent, label):
        class_all = len(np.unique(label))
        label = label.reshape(label.shape[0] * label.shape[1])
        represent = represent.reshape(represent.shape[0],represent.shape[1] * represent.shape[2])
        kmeans = KMeans(n_clusters=1)
        memory_now = represent
        for i in range(class_all):
            kmeans.fit(represent[i, :, ])
            class_meomry = kmeans.cluster_centers_
            memory_now[i, :, :] = class_meomry