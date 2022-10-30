import imageio
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os

# torch.manual_seed(1)  # reproducible
#
# transform = transforms.Compose([
#     transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
# ])
'''NPY数据格式'''


class MyDataset(Dataset):
    def __init__(self, image_path,label_path):

        self.image_path = image_path
        self.label_path = label_path

        self.images = os.listdir(image_path)  #所有文件名
        self.labels = os.listdir(label_path)

    def __getitem__(self, index):

        img_name = self.images[index]

        img_path = os.path.join(self.image_path,img_name)
        label_path = os.path.join(self.label_path,img_name)

        image = imageio.imread(img_path)
        image = np.transpose(image,(2,0,1))

        label = imageio.imread(label_path)


        return torch.tensor(image), torch.tensor(label)
        # return ldct, hdct  # 返回数据还有标签

    def __len__(self):
        return len(self.images) # 返回数据的总个数


def main():
    image_path = r"C:\Users\Administrator\Desktop\New_Orson\Train_Npy\imagesDataset_.npy"
    label_path = r"C:\Users\Administrator\Desktop\New_Orson\Train_Npy\labelsDataset_.npy"
    mydataset = MyDataset(image_path, label_path)

    # dataset = MyDataset('.\data_npy\img_covid_poisson_glay_clean_BATCH_64_PATS_100.npy')
    data = DataLoader(mydataset, batch_size=8, shuffle=True, pin_memory=True)
    print(data[0])


if __name__ == '__main__':
    main()