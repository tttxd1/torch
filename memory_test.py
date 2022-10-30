from sklearn.cluster import KMeans
import numpy as np
import torch
import torch.nn as nn


def memory_module(input, numclass, epoches, step):
    memory = np.random.randint(1, size=(numclass, input.shape[-1], input.shape[-1]))
    memory_now = input
    kmeans = KMeans(n_clusters=1)

    for i in range(numclass):
        input = input.detach()
        kmeans.fit(input[i, :, ])
        class_meomry = kmeans.cluster_centers_
        memory_now[i, :, :] = torch.tensor(class_meomry)

    m = ((1 - step / epoches) ** 0.9) * (0.9 - 0.9 / 100) + 0.9 / 100

    memory = m * memory + m * memory_now

    return memory

class Net(nn.Module):
    def __init__(self, numclass=7, in_channels=4):
        super(Net, self).__init__()

        self.start_cov = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.last_cov = nn.Conv2d(16, numclass, kernel_size=3, padding=1)
        self.softmax = nn.Softmax()
        self.numclass = numclass
        pass

    def forward(self, x, epoches, step):
        x = self.start_cov(x)
        x = self.last_cov(x)
        x = self.softmax(x)

        memory = memory_module(x, self.numclass, epoches, step)


        return x , memory



a = np.random.randint(5,size=(4,6,6))
a = torch.Tensor(a)
net = Net()
epoches = 300
step = 1
output , me= net(a, epoches, step)
print(output.shape)
print(me.shape)



# memory =  np.random.randint(5,size=(7, 6, 6))
# kmeans=KMeans(n_clusters=1)
# output = output.detach().numpy()
# memory_now = output
# for i in range(7):
#     kmeans.fit(output[i,:,])
#     class_meomry = kmeans.cluster_centers_
#     memory_now[i,:,:] = class_meomry
#
# epoches = 500
# i = 1
# m =  ((1 -i/epoches ) ** 0.9 ) * (0.9 - 0.9/100) + 0.9/100
#
# memory = m * memory + m * memory_now
# print(memory.shape)
#
#
#
#
# checkpoint = {
#    'memory': memory,
#    'model': net.state_dict(),
#  }
# save_path = r"C:\Users\224\Desktop\论文\记忆\1.pth"
# torch.save(checkpoint,save_path)
#
# net = Net()
# resume_dir = r"C:\Users\224\Desktop\论文\记忆\1.pth"
# checkpoint = torch.load(resume_dir)
# model_ckt = checkpoint["model"]
# meomry = checkpoint["memory"]
#
# net.load_state_dict(model_ckt)
# print(meomry.shape)

# kmeans=KMeans(n_clusters=1)
# kmeans.fit(a)
# y_kmeans=kmeans.predict(a)  #聚类类别结果
# print(y_kmeans)
# centroids=kmeans.cluster_centers_  #聚类中心点
# print(centroids)




