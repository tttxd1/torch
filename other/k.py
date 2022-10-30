import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import random

#构造的训练集#
x = torch.rand(100,28,28)
y = torch.randn(100,28,28)
x = torch.cat((x,y),dim=0)
label =[1] *100 + [0]*100
label = torch.tensor(label,dtype=torch.long)

index = [i for i in range(len(x))] 			# 打乱数据
random.shuffle(index)
x = x[index]
label = label[index]

# Model
class Net(nn.Module):
	...
# DataSet
class TraindataSet(Dataset):
    def __init__(self,train_features,train_labels):
        pass

# k折划分
def get_k_fold_data(k, i, X, y):
    # 返回第i折交叉验证时所需要的训练和测试数据，分开放，X_train为训练数据，X_test为验证数据
    assert k > 1
    fold_size = X.shape[0] // k  # 每份的个数:数据总条数/折数（向下取整）

    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)  # slice(start,end,step)切片函数 得到测试集的索引

        X_part, y_part = X[idx, :], y[idx]  # 只对第一维切片即可
        if j == i:  # 第i折作test
            X_test, y_test = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)  # 其他剩余折进行拼接 也仅第一维
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_test, y_test


loss_func = nn.CrossEntropyLoss()  # 声明Loss函数

#单折训练过程
def train(net, X_train, y_train, X_test, y_test, num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []  # 存储train_loss,test_loss
    # 最不一样的地方就是在这里才进行将数据处理成数据集便于加载数据
    dataset = TraindataSet(X_train, y_train)
    train_iter = DataLoader(dataset, batch_size, shuffle=True)

    # Adam优化算法
    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        for X, y in train_iter:  # 分批训练
            output = net(X)
            loss = loss_func(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        ### 得到每个epoch的 loss 和 accuracy
    # 返回这一折的结果
    return train_ls, test_ls


def k_fold(k, X_train, y_train, num_epochs=3, learning_rate=0.001, weight_decay=0.1, batch_size=5):
    train_loss_sum, test_loss_sum = 0, 0
    train_acc_sum, test_acc_sum = 0, 0

    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)  # 获取第i折交叉验证的训练和验证数据
        net = Net()  # 每个K都要实例化新模型
        # 返回这一折的训练结果
        train_ls, test_ls = train(net, *data, num_epochs, learning_rate, \
                                  weight_decay, batch_size)
        ...  # 对这一折的训练结果进行处理
        ...
    ...  # 对总的结果进行平均输出
    ...


k_fold(10, x, label)  # k=10,十折交叉验证