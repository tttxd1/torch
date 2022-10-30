# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 09:00:44 2019

@author: zhangyonghui
"""

# ===================================================================================================

import numpy as np
import imageio
import cv2
from datetime import datetime
import glob
import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from Unet import UNet

model = UNet(num_classes=7)
model.load_state_dict(torch.load( r"E:\yqj\try\code\torch\Train\save_model\UNet\256_max\350-0.01595.pth",map_location = torch.device('cpu')))

# ===================================================================================================

# 超参数
img_size = 1024
classes = 16
LR = 1e-4
input_sizes = (img_size, img_size, 4)


# ===================================================================================================


def makedirs(dir_path):
    """
    本函数实现以下功能：
        1、创建路径文件夹
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print('{}: Folder creation successful: {}'.format(datetime.now().strftime('%c'), dir_path))
    else:
        print('{}: Folder already exists: {}'.format(datetime.now().strftime('%c'), dir_path))

    return dir_path


# ===================================================================================================

def estimate(y_label, y_pred):
    """
    本函数实现以下功能：
        1、掩膜
        2、计算准确率
    """
    # 掩膜
    # y_pred[y_label == 0] = 0

    # 准确率
    acc = np.mean(np.equal(y_label, y_pred) + 0)

    return acc, y_pred


# ===================================================================================================

def model_predict(model, img_data, lab_data, img_size):
    """
    本函数实现以下功能：
        1、对于一幅高宽较大的图像，实现分块预测，每块的大小是参数 img_size

    @parameter:
        model: 模型参数
        img_data：需要预测的图像数据
        lab_data：需要育德的图像的标签
        img_size：预测图像块的大小（不等于 img_data 的大小）
    """
    # 获取预测图像的 shape
    row, col, dep = img_data.shape

    # 为了查看信息，没什么用
    if row % img_size != 0 or col % img_size != 0:
        print('{}: Need padding the predict image...'.format(datetime.now().strftime('%c')))
        # 计算填充后图像的 hight 和 width
        padding_h = (row // img_size + 1) * img_size
        padding_w = (col // img_size + 1) * img_size
    else:
        print('{}: No need padding the predict image...'.format(datetime.now().strftime('%c')))
        # 不填充后图像的 hight 和 width
        padding_h = (row // img_size) * img_size
        padding_w = (col // img_size) * img_size

    # 初始化一个 0 矩阵，将图像的值赋值到 0 矩阵的对应位置
    padding_img = np.zeros((padding_h, padding_w, dep), dtype='float32')
    padding_img[:row, :col, :] = img_data[:row, :col, :]

    # 初始化一个 0 矩阵，用于将预测结果的值赋值到 0 矩阵的对应位置
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
            img_data_ = padding_img[i:i + img_size, j:j + img_size, :]
            img_data_ = img_data_[np.newaxis, :, :, :]

            # 预测，对结果进行处理
            y_pre = model(torch.tensor(img_data_))
            y_pre = torch.squeeze(y_pre, axis=0)
            y_pre = torch.argmax(y_pre, axis=-1)
            y_pre = y_pre.numpy().astype('uint8')

            # 将预测结果的值赋值到 0 矩阵的对应位置
            padding_pre[i:i + img_size, j:j + img_size] = y_pre[:img_size, :img_size]

            count += 1  # 每预测一块就+1

            print('\r{}: Predited {:<5d}({:<5d})'.format(datetime.now().strftime('%c'), count,
                                                         int((padding_h / img_size) * (padding_w / img_size))), end='')

    # 计算准确率
    acc, y_pred = estimate(lab_data, padding_pre[:row, :col] + 1)

    return acc, y_pred


# =========================================================================================

if __name__ == '__main__':
    """
    主函数
    """
    """
    加载图像信息
    """
    # 预测图像和标签
    # pre_name = 'qilianyu_GF2_clip3.tif'
    # lab_name = 'shanhujiao_label_13classes.tif'
    pre_name = 'image1.tif'
    lab_name = 'new_LHJ.tif'
    # 加载预测图像
    # img_path = os.path.join(os.getcwd(), 'lable','lingyangjiao','11','image', pre_name)
    img_path = r"E:\yqj\code\GF\Code\lable\langhuajiao\image\image1.tif"
    if not os.path.exists(img_path):
        print('{}: Do not find the image: {}'.format(datetime.now().strftime('%c'), img_path))
    img_data = imageio.imread(img_path)
    img_data = np.float32(img_data)

    # 最大最小归一化
    B1, B2, B3, B4 = cv2.split(img_data)
    B1_normalization = ((B1 - np.min(B1)) / (np.max(B1) - np.min(B1)) * 1).astype('float32')
    B2_normalization = ((B2 - np.min(B2)) / (np.max(B2) - np.min(B2)) * 1).astype('float32')
    B3_normalization = ((B3 - np.min(B3)) / (np.max(B3) - np.min(B3)) * 1).astype('float32')
    B4_normalization = ((B4 - np.min(B4)) / (np.max(B4) - np.min(B4)) * 1).astype('float32')
    image_new_data = cv2.merge([B1_normalization, B2_normalization, B3_normalization, B4_normalization])

    # 加载预测标签
    img_name = pre_name.split('.tif')[0]
    # lab_path = os.path.join(os.getcwd(),  'lable','lingyangjiao','11','label', lab_name)
    lab_path = r"E:\yqj\code\GF\Code\lable\langhuajiao\label\new_LHJ.tif"
    if not os.path.exists(lab_path):
        print('{}: Do not find the label: {}'.format(datetime.now().strftime('%c'), lab_name))
    lab_data = imageio.imread(lab_path)

    """
    加载模型信息
    """

    # ===================================================================================================
    '''
        加载模型信息
    '''
    model_name_list = ['UNet', 'MultiResUnet', 'PSPNet', 'SegNet', 'Deeplabv3P', 'OurNet', 'UNet2', 'ENet']
    model_name = model_name_list[0]

    # 导入模型
    #     print('==========================Start {}!======================================='.format(model_name))
    # elif model_name == model_name_list[2]:
    #     model = mdoel_segnet.SegNet(input_size=input_sizes, num_class=classes, model_summary=True)
    #     print('==========================Start {}!======================================='.format(model_name))
    # elif model_name == model_name_list[3]:
    #     model = model_Deeplabv3P.Deeplabv3(input_size=input_sizes, classes=classes, LR=LR, model_summary=True)
    #     print('==========================Start {}!======================================='.format(model_name))
    # elif model_name == model_name_list[4]:
    #     model = model_ourNet.ourNet(input_size=input_sizes, num_class=classes, model_summary=True)
    #     print('==========================Start {}!======================================='.format(model_name))
    # elif model_name == model_name_list[5]:
    #     model = model_unet2.unet2(input_size=input_sizes, num_class=classes, model_summary=True)
    #     print('==========================Start {}!======================================='.format(model_name))
    # elif model_name == model_name_list[6]:
    #     model = model_enet.ENET(input_size=input_sizes, num_classs=classes, model_summary=True)
    #     print('==========================Start {}!======================================='.format(model_name))


    # ===================================================================================================

    # 预测结果的存放目录
    y_pre_savepath = makedirs(os.path.join(os.getcwd(), 'lable', '_dayitong', 'result', img_name, model_name))

    # 获取模型列表
    # model_list = glob.glob(os.path.join(os.getcwd(), 'save_model', model_name, '*.hdf5'))
    model_list = glob.glob(os.path.join(r'E:\yqj\code\GF\Code\lable\_dayitong\model\all', '*.hdf5'))
    print('{}: Find {} model\n'.format(datetime.now().strftime('%c'), len(model_list)))

    for i in range(len(model_list)):
        # 获取模型名字
        model_name = (model_list[i].split('\\')[-1]).split('.hdf5')[0]
        print('\n{}: Statr No.{:<3d} model, the name is {}'.format(datetime.now().strftime('%c'), i + 1, model_name))

        # 加载模型参数
        model.load_weights(model_list[i])

        # 预测结果
        accuracy, predict_result = model_predict(model, image_new_data, lab_data, img_size)
        print(
            '\n{}: The accuracy rate is {:<0.5f}, the model name is {}'.format(datetime.now().strftime('%c'), accuracy,
                                                                               model_name))



        # 保存预测结果
        # imageio.imwrite(os.path.join(y_pre_savepath, model_name+'_acc-'+str(round(accuracy, 4))+'.tif'), predict_result)
        print('{}: Predict the success of image {}\n'.format(datetime.now().strftime('%c'), pre_name))

# =========================================================================================





