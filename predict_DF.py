import os.path
import torch.nn
from torchvision import transforms
import imageio
from PIL import Image
import numpy as np
import cv2
from datetime import datetime
from Model.UNet.unet import UNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def estimate(y_label, y_pred):
    y_pred[y_label==0]=0
    # 准确率
    acc = np.mean(np.equal(y_label, y_pred) + 0)

    return acc, y_pred

def model_predict(model, img_data, lab_data, img_size):

    row, col, dep = img_data.shape

    if row % img_size != 0 or col % img_size != 0:
        print('{}: Need padding the predict image...'.format(datetime.now().strftime('%c')))
        # 计算填充后图像的 hight 和 width
        padding_h = (row // img_size + 1) *img_size
        padding_w = (col // img_size + 1) *img_size
    else:
        print('{}: No need padding the predict image...'.format(datetime.now().strftime('%c')))
        # 不填充后图像的 hight 和 width
        padding_h = (row // img_size) *img_size
        padding_w = (col // img_size) *img_size

    # 初始化一个 0 矩阵，将图像的值赋值到 0 矩阵的对应位置
    padding_img = np.zeros((padding_h, padding_w, dep), dtype='float32')
    padding_img[:row, :col, :] = img_data[:row, :col, :]

    #初始化一个 0 矩阵，用于将预测结果的值赋值到 0 矩阵的对应位置
    padding_pre = np.zeros((padding_h, padding_w), dtype='uint8')

    # 对 img_size * img_size 大小的图像进行预测
    count = 0  # 用于计数
    for i in list(np.arange(0, padding_h, img_size)):
        if (i + img_size) > padding_h:
            continue
        for j in list(np.arange(0, padding_w, img_size)):
            if (j + img_size) > padding_w:
                continue

            # 取 img_size 大小的图像，在第一维添加维度，变成四维张量，用于模型预测
            img_data_ = padding_img[i:i+img_size, j:j+img_size, :]
            toTensor = transforms.ToTensor()
            img_data_ = toTensor(img_data_)
            img_data_ = img_data_[np.newaxis, :, :, :]
            # img_data_ = np.transpose(img_data_, (0, 3, 1, 2))

            # 预测，对结果进行处理
            y_pre = model.forward(img_data_)
            # y_pre = model.predict(img_data_)
            y_pre = np.squeeze(y_pre, axis = 0)
            y_pre = torch.argmax(y_pre, axis = 0)
            # y_pre = y_pre.astype('uint8')

            # 将预测结果的值赋值到 0 矩阵的对应位置
            padding_pre[i:i+img_size, j:j+img_size] = y_pre[:img_size, :img_size]

            count += 1  # 每预测一块就+1


            print('\r{}: Predited {:<5d}({:<5d})'.format(datetime.now().strftime('%c'), count, int((padding_h/img_size)*(padding_w/img_size))), end='')

    # 计算准确率
    acc, y_pred = estimate(lab_data, padding_pre[:row, :col]+1)

    return acc, y_pred

#参数
num_class = 16
image_size = 256
imagePath = r"E:\yqj\try\code\torch\Train\test\1_2_5 (1).tif"
labelPath = r"E:\yqj\try\code\torch\Train\test\1_2_5.tif"
modelPath = r"E:\yqj\try\code\torch\Train\save_model\UNet\400-0.03467.pth"
savePath = r"E:\yqj\try\测试\1028.tif"

image = imageio.imread(imagePath)
label = imageio.imread(labelPath)
# image = Image.open(imagePath)
# image = np.array(image)
# label = Image.open(labelPath)
# label = np.array(label)
# B1, B2, B3, B4 = cv2.split(image)
# B1_normalization = ((B1 - np.min(B1)) / (np.max(B1) - np.min(B1)) * 1).astype('float32')
# B2_normalization = ((B2 - np.min(B2)) / (np.max(B2) - np.min(B2)) * 1).astype('float32')
# B3_normalization = ((B3 - np.min(B3)) / (np.max(B3) - np.min(B3)) * 1).astype('float32')
# B4_normalization = ((B4 - np.min(B4)) / (np.max(B4) - np.min(B4)) * 1).astype('float32')
# image = cv2.merge([B1_normalization, B2_normalization, B3_normalization, B4_normalization])
#加载模型
model = UNet(num_classes=num_class)
model.load_state_dict(torch.load(modelPath, map_location = torch.device('cpu')))
model.eval()

# output = model.forward(image)
acc, output = model_predict(model, image, label, img_size=image_size)
# output = output.numpy()
# output = output.argmax(dim = 0)
print(f"准确率： {acc}")
imageio.imwrite( savePath,output)






