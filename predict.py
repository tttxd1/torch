import os.path
import torch.nn
import imageio
import numpy as np
import cv2
from datetime import datetime
from Model.UNet.unet import UNet
import glob
from utils.utils import Logger
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

now = datetime.now()
now = str(now.month) + '_' + str(now.day) + '_' + str(now.hour) + '_' + str(now.minute)

def estimate(y_label, y_pred):
    y_pred[y_label==0]=0
    # 准确率
    acc = np.mean(np.equal(y_label, y_pred))

    return acc, y_pred

def readimage(dir):
    images_path_list = glob.glob(os.path.join(dir, '*.tif'))
    return images_path_list



def model_predict(model, img_data, lab_data, img_size):
    row, col, dep = img_data.shape

    if row % img_size != 0 or col % img_size != 0:
        # 计算填充后图像的 hight 和 width
        padding_h = (row // img_size + 1) * img_size
        padding_w = (col // img_size + 1) * img_size
    else:
        # 不填充后图像的 hight 和 width
        padding_h = (row // img_size) * img_size
        padding_w = (col // img_size) * img_size

    # 初始化一个 0 矩阵，将图像的值赋值到 0 矩阵的对应位置
    padding_img = np.zeros((padding_h, padding_w, dep), dtype='float32')
    padding_img[:row, :col, :] = img_data[:row, :col, :]

    # 初始化一个 0 矩阵，用于将预测结果的值赋值到 0 矩阵的对应位置
    padding_pre = np.zeros((padding_h, padding_w), dtype='uint8')
    # padding_pre = torch.tensor(padding_pre)
    # padding_pre = padding_pre.to(device)

    # 对 img_size * img_size 大小的图像进行预测
    count = 0  # 用于计数
    for i in list(np.arange(0, padding_h, img_size)):
        if (i + img_size) > padding_h:
            continue
        for j in list(np.arange(0, padding_w, img_size)):
            if (j + img_size) > padding_w:
                continue

            # 取 img_size 大小的图像，在第一维添加维度，变成四维张量，用于模型预测
            img_data_ = padding_img[i:i + img_size, j:j + img_size, :]
            img_data_ = img_data_[np.newaxis, :, :, :]
            img_data_ = np.transpose(img_data_, (0, 3, 1, 2))
            img_data_ = torch.from_numpy(img_data_)

            # 预测，对结果进行处理
            y_pre = model(img_data_.to(device))
            # y_pre = model.predict(img_data_)
            y_pre = np.squeeze(y_pre, axis=0)
            y_pre = torch.argmax(y_pre, axis=0)
            # y_pre = y_pre.astype('uint8')

            # 将预测结果的值赋值到 0 矩阵的对应位置
            padding_pre[i:i + img_size, j:j + img_size] = y_pre[:img_size, :img_size]

            count += 1  # 每预测一块就+1

    # 计算准确率
    acc , y_pre = estimate(lab_data, padding_pre[:row, :col] )

    return acc,y_pre

#参数
num_class = 7
image_size = 512
modelname = "UNet"
imagedir = r"G:\数据集\连云港\NewDataSet_5.18\pre_image_vaild"
labeldir = r"G:\数据集\连云港\NewDataSet_5.18\coastline_classify_label_vaild"
modelPath = r"E:\yqj\try\code\torch\Train\save_model\UNet\512\600-0.0.pth"#r"E:\yqj\try\code\torch\Train\save_model\UNet\510-0.01232.pth"#
savePath = r"E:\yqj\try\测试\coastline" + "\\" + modelname
log_path = r"E:\yqj\try\code\torch\logs\coastline_eva" + "\\"+ now+ ".log"

if os.path.exists(savePath) == False:
    os.makedirs(savePath)

f = open(log_path, 'w')
f.close()

log = Logger(log_path, level='debug')
log.logger.info('Start! Train image size  ' + str(image_size))
log.logger.info('modelPath: ' + modelPath)

imagelist = readimage(imagedir)
labellist = readimage(labeldir)

acc_all = 0
for i in range(len(imagelist)):
        image = imageio.imread(imagelist[i])
        label = imageio.imread(labellist[i])
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
        acc_all += acc
        # output = output.numpy()
        # output = output.argmax(dim = 0)
        # print(f"{i + 2016}的准确率： {acc}")
        log.logger.info(f"{i + 2016}的准确率： {acc}")
        save = savePath + "\\" + str(i+16) + ".tif"
        imageio.imwrite( save,output)


# print(f"平均准确率： {acc_all/len(imagelist)}")
log.logger.info(f"平均准确率： {acc_all/len(imagelist)}")



