import os.path
import numpy as np
from datetime import datetime

import torch.nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter #tensorboard  --> 可视化
from torch.optim.lr_scheduler import ExponentialLR,CosineAnnealingLR
from utils.dataset.TT_Dataset import MyDataset
from Model.UNet.unet_coastline import UNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def estimate(y_label, y_pred):
    y_pred[y_label==0]=0
    # 准确率
    acc = np.mean(np.equal(y_label, y_pred) + 0)

    return acc

#参数
num_classes = 7
epochs = 200
batch_size = 1
modelname = "UNet"

#可视化
now = datetime.now()
now = str(now.month) + '_' + str(now.day) + '_' + str(now.hour) + '_' + str(now.minute)

tensorboardPath = os.path.join(os.getcwd(), 'logs',modelname,now)  #可视化文件所在的文件夹
if os.path.exists(tensorboardPath) == False:
    os.makedirs(tensorboardPath)
writer = SummaryWriter(tensorboardPath)

#数据处理
imagePath = r"E:\yqj\try\code\torch\Train\Data\coastline\train\image"
labelPath = r"E:\yqj\try\code\torch\Train\Data\coastline\train\label\class"

#构建数据集
trainDataset = MyDataset(imagePath, labelPath)
trainDatasetloader = DataLoader(trainDataset, batch_size)
trainLen = len(trainDataset)

#定义模型
model = UNet(num_classes=num_classes).to(device)
total = sum([param.nelement() for param in model.parameters()])
print("Number of parameter: %.2fM" % (total / 1e6))

#损失函数 优化器
loss = CrossEntropyLoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=0.05, betas=(0.9, 0.95))
scheduler = CosineAnnealingLR(optimizer, T_max=50) #学习率调整
print(f"优化器参数： {optimizer}")

#训练
for epoch in range(epochs):
    print(f"----------------------------------------------epoch: {epoch}----------------------------------------------")
    total_loss = 0
    total_acc = 0
    num = 0
    for i, data in enumerate(trainDatasetloader):
        img, lab, name = data
        img = img.to(device)
        lab = lab.to(device)

        #梯度清零
        optimizer.zero_grad()

        output, memory = model(img)
        model_loss = loss(output, lab.long())

        #loss回传 优化器更新参数
        model_loss.backward()
        optimizer.step()

        lr = optimizer.param_groups[0]["lr"]     #当前学习率
        print("\r train: epoch: {}, step: {}/{}, lr: {}, loss: {}".format(epoch, i, trainLen, lr, round(float(model_loss), 8)), end='')
        total_loss += model_loss                 #每一步的loss求和

        #可视化的参数
        writer.add_scalar('Train Step Loss ', model_loss,epoch * len(trainDataset) + i )
        writer.add_scalar('Train Step Lr ', lr, epoch * len(trainDataset) + i)

    #每一个epoch的平均loss
    epoch_loss = total_loss * 1.0 / len(trainDataset)
    # epoch_acc = total_acc * 1.0 /len(valDataset)
    scheduler.step()

    writer.add_scalar('Train Epoch Loss', epoch_loss, epoch + 1 )
    print("\r epoch: {}, epoch_loss: {}".format(epoch, epoch_loss), end='')
    #保存模型
    save_name = str(epoch+1) + '-' +str(round(float(epoch_loss), 5)) + ".pth"  #模型名称
    savepath = os.path.join(os.getcwd(), 'Train', "save_model", modelname)     #模型所在文件夹

    Path = os.path.join(savepath, save_name)
    if os.path.exists(savepath) == False:
        os.makedirs(savepath)

    checkpoint = {
        'memory': memory,
        'model': model.state_dict(),
    }
    torch.save(checkpoint, Path)

    # #前190个epoch： 每20个epoch保存一次
    # if((epoch + 1) % 20 == 0 and (epoch + 1) < 190):
    #     torch.save(model.state_dict(),Path)
    # # 最后10个epoch： 每个epoch保存一个
    # elif((epoch + 1) > 190):
    #     torch.save(model.state_dict(), Path)






