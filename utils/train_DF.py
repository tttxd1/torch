import torch
from torch.utils.data import Dataset,DataLoader
from Models import Unet,MyDataset,new_unet,Dataset_nonpy,UNet2
from torch.nn import MSELoss,CrossEntropyLoss
from torch.optim import SGD,Adam

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#超参
learning_rate = 1e-4
epochs = 200
batch = 8


#数据集
image_path = r"C:\Users\Administrator\Desktop\New_Orson\new_orson_B2345\cut_train_images"
label_path = r"C:\Users\Administrator\Desktop\New_Orson\new_orson_B2345\cut_train_labels"

mydataset = Dataset_nonpy.MyDataset(image_path,label_path)
data_loader = DataLoader(dataset=mydataset,batch_size=batch, shuffle=True, pin_memory=True)

#定义网络
net = UNet2.UNet(7).to(device)

#损失函数
loss_ = CrossEntropyLoss() #MSELoss()

#优化器
#optimizer = SGD(net.parameters(),lr = learning_rate,momentum=0.9)
optimizer = Adam(net.parameters(),lr = learning_rate)
#回调
#模型编译

#模型训练
for e in range(epochs):
    print("--------------epoch:{}/{}------------------".format(e+1,epochs))

    # net.train()
    epoch_loss = 0
    step = 0
    lowest_loss = 10

    for data in data_loader:
        #梯度清零
        optimizer.zero_grad()

        step+=1
        img,label = data

        output = net(img)
        loss = loss_(output, label.long())
        epoch_loss += loss

        loss.backward()
        optimizer.step()

        print("step:{},loss:{}".format(step,loss))
    avg_loss = epoch_loss/step
    print("epoch:{},loss:{}".format(e+1,avg_loss))


    #保存模型，损失降低保存模型参数
    if avg_loss < lowest_loss:
        torch.save(net.state_dict(),'weights_1015_2026.pth')
        print("epoch{},saving models...".format(e))

    # net.eval()
    # #val_loss_sum = 0.0
    # val_metric_sum = 0.0
    # val_step = 1
    #
    # for val_step, (features, labels) in enumerate(dl_valid, 1):
    #     # 关闭梯度计算
    #     with torch.no_grad():
    #         pred = net(features)
    #         #val_loss = loss_func(pred, labels)
    #         val_metric = metric_func(labels, pred)
    #     #val_loss_sum += val_loss.item()
    #     val_metric_sum += val_metric.item()