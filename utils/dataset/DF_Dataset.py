import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms import  ToTensor
from torch.utils.data import Dataset, DataLoader

# torch.manual_seed(1)  # reproducible
#
# transform = transforms.Compose([
#     transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
# ])
'''NPY数据格式'''


class MyDataset(Dataset):
    def __init__(self, image_data,label_path):
        self.image_data = np.load(image_data)  # 加载npy数据
        self.label_data = np.load(label_path)
        # self.transforms = transform  # 转为tensor形式
        self.totensor = ToTensor()

    def __getitem__(self, index):

        # 读取每一个npy的数据
        image = self.image_data[index, :, :, :]
        image = np.transpose(image,(2,0,1))

        label = self.label_data[index, :, :]
        label = np.transpose(label, (2, 0, 1))


        return image, label
        # return ldct, hdct  # 返回数据还有标签

    def __len__(self):
        return self.image_data.shape[0]  # 返回数据的总个数


def main():
    image_path = r"E:\yqj\code\GF\Code\lable\huaguangjiao\train\Train_Npy\imagesDataset_.npy"
    label_path = r"E:\yqj\code\GF\Code\lable\huaguangjiao\train\Train_Npy\labelsDataset_.npy"
    mydataset = MyDataset(image_path, label_path)

    # dataset = MyDataset('.\data_npy\img_covid_poisson_glay_clean_BATCH_64_PATS_100.npy')
    Data = DataLoader(mydataset, batch_size=8, shuffle=True, pin_memory=True)
    for i, data in enumerate(Data):
        img, lab = data
        print(img.shape)
        print(lab.shape)



if __name__ == '__main__':
    main()