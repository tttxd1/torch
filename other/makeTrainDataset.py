# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 16:06:31 2019

@author: zhangyonghui
"""

import imageio
import os
import numpy as np
import cv2


# 根目录为当前文件所在目录
root = os.getcwd()

#====================================================================================================================================

def cut_image(image, image_name, block_size, stride, save_path):

    row = image.shape[0]
    col = image.shape[1]
    dep = image.shape[2]
    # if row % block_size != 0 or col % block_size != 0:
    print('Need padding the image...')
    # 计算填充后图像的 hight 和 width
    padding_h = (row // block_size + 1) *block_size
    padding_w = (col // block_size + 1) *block_size
    # 初始化一个 0 矩阵，将图像的值赋值到 0 矩阵的对应位置
    padding_img = np.zeros((padding_h, padding_w, dep), dtype='float32')
    padding_img[:row, :col, :] = image[:row, :col, :]

    row_num = 0
    for i in list(np.arange(0, row, stride)):
        row_num += 1

        if (i + block_size) > row:
            continue

        col_num = 0
        for j in list(np.arange(0, col, stride)):
            col_num += 1

            if (j + block_size) > col:
                continue

            block = np.array(padding_img[i: i+block_size, j: j+block_size, :])
            block_name = image_name + '_' + str(int(row_num)) + '_' + str(int(col_num)) + '.tif'

            imageio.imwrite(os.path.join(save_path, block_name), block)


def cut_label(image, image_name, block_size, stride, save_path):

    row = image.shape[0]
    col = image.shape[1]
    
    # if row % block_size != 0 or col % block_size != 0:
    print('Need padding the image...')
    # 计算填充后图像的 hight 和 width
    padding_h = (row // block_size + 1) *block_size
    padding_w = (col // block_size + 1) *block_size
    #初始化一个 0 矩阵，用于将预测结果的值赋值到 0 矩阵的对应位置
    padding_pre = np.zeros((padding_h, padding_w), dtype='uint8')
    padding_pre[:row, :col] = image[:row, :col]
    
    row_num = 0
    for i in list(np.arange(0, row, stride)):
        row_num += 1

        if (i + block_size) > row:
            continue

        col_num = 0
        for j in list(np.arange(0, col, stride)):
            col_num += 1

            if (j + block_size) > col:
                continue

            block = np.array(padding_pre[i: i+block_size, j: j+block_size])
#            if np.min(block) == 0:
#                print('ok')
            block_name = image_name + '_' + str(int(row_num)) + '_' + str(int(col_num)) + '.tif'

            imageio.imwrite(os.path.join(save_path, block_name), block)

#====================================================================================================================================

"""
生成彩色图像，为了标记样本
"""

GF_name_list = list(np.arange(30)+1)

# image 文件路径
# image_path = os.path.join(root, 'lable','xisha','12','train','train_image')
image_path = r"E:\yqj\try\code\torch\Train\Data\coastline\image\train"

# label 文件路径
# label_path = os.path.join(root,'lable','xisha','12','train','train_label')
label_path = r"E:\yqj\try\code\torch\Train\Data\coastline\label\coastline_classify_label_train"

for i in range(len(GF_name_list)):

    # 整幅图像的数据和标签
    # image_data = imageio.imread(os.path.join(image_path,'lingyangjiao.tif'))
    # label_data = imageio.imread(os.path.join(label_path ,'lingyangjiao_lable.tif'))
    image_data = imageio.imread(os.path.join(image_path, str(GF_name_list[i])+'.tif'))
    label_data = imageio.imread(os.path.join(label_path, str(GF_name_list[i])+'.tif'))

    label_data = label_data - 1

    # B1, B2, B3, B4 = cv2.split(image_data)
    # B1_normalization = ((B1 - np.min(B1)) / (np.max(B1) - np.min(B1)) * 1).astype('float32')
    # B2_normalization = ((B2 - np.min(B2)) / (np.max(B2) - np.min(B2)) * 1).astype('float32')
    # B3_normalization = ((B3 - np.min(B3)) / (np.max(B3) - np.min(B3)) * 1).astype('float32')
    # B4_normalization = ((B4 - np.min(B4)) / (np.max(B4) - np.min(B4)) * 1).astype('float32')
    # image_data = cv2.merge([B1_normalization, B2_normalization, B3_normalization, B4_normalization])

    # 创建切割后的'image'保存文件夹
    # train_img_path = os.path.join(root, 'lable', 'small', 'train','cut_train_images')
    train_img_path = r"E:\yqj\try\code\torch\Train\Data\coastline\train_256\image"
    if not os.path.exists(train_img_path):
        os.makedirs(train_img_path)

    # 创建切割后的'label'保存文件夹
    # train_lab_path = os.path.join(os.getcwd(), 'lable', 'small', 'train','cut_train_labels')
    train_lab_path = r"E:\yqj\try\code\torch\Train\Data\coastline\train_256\label\class"
    if not os.path.exists(train_lab_path):
        os.makedirs(train_lab_path)
        
    # 切割
    cut_image(image_data, str(GF_name_list[i]), 256, 250, train_img_path)
    cut_label(label_data, str(GF_name_list[i]), 256, 250, train_lab_path)










