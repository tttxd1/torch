from torch.nn.functional import  one_hot

import cv2, imageio
# import matplotlib.pyplot as plt
from skimage import measure, data, color, io, morphology
from skimage import measure, data, color, io
# from skimage import *
from osgeo import gdal, osr
from sys import argv
import tkinter as tk
from tkinter import *
import numpy as np
import os, glob


def InitList(x, y):
    # 创建x * y的二维空数组
    array = [([0] * y) for i in range(x)]
    return array


# array：分类结果数组；
# radius:1;
# x,y:岸线第n个像素的坐标
def GetListByCoord(array, radius, x, y):
    # 根据半径来确定数组的行数和列数
    row_col = 2 * radius + 1
    # 初始化结果数组
    # result现在是3*3的空数组
    result = InitList(row_col, row_col)

    # 获取传入的array的行数和列数
    arrayRow, arrayCol = len(array), len(array[0])
    # judge数组
    judge = [[0 for j in range(row_col)] for i in range(row_col)]
    # 坐标x、y的值即为结果数组的中心，依此为偏移
    flag = 3
    # 改成右下角
    for i in range(result.__len__()):
        for j in range(result.__len__()):
            # 是否越界
            # print(i)
            # i + x - radius，j + y - radius：中心点(左上角点？)
            # 判断点是否越界
            if (i + x - radius < 0 or j + y - radius < 0 or i + x - radius >= arrayRow or j + y - radius >= arrayCol):
                continue
            # elif
            elif array[i + x - radius][j + y - radius] != 7:
                judge[i][j] = array[i + x - radius][j + y - radius]
                flag = array[i + x - radius][j + y - radius]


            elif array[i + x - radius][j + y - radius] == 7:
                judge[i][j] = flag

                # line_result[x][y] = array[i + x - radius][j + y - radius]
                # return

            # line_result[x][y] = 7
    j = np.array(judge)
    j = j.flatten()
    value = np.argmax(np.bincount(j))
    # value = np.where(j == np.max(j[:]))
    # a,b = value[0][0],value[1][0]
    line_result[x][y] = value
    return


def remove_noise(filename):
    img = filename
    img = img / 255
    n_class = 2
    area_threshold = 500
    result = one_hot(img, num_classes=n_class)  # 转为one-hot
    result1 = []
    result1.append(result)
    result2 = np.array(result1)
    for i in range(n_class):
        # 去除小物体
        result2[:, :, :, i] = morphology.remove_small_objects(result2[:, :, :, i] == 1, min_size=area_threshold,
                                                              connectivity=1, in_place=True)
        # 去除孔洞
        result2[:, :, :, i] = morphology.remove_small_holes(result2[:, :, :, i] == 1, area_threshold=area_threshold,
                                                            connectivity=1, in_place=True)
    # 获取最终label
    result4 = np.squeeze(np.argmax(result2, axis=-1).astype(np.uint8))
    result4 = result4 * 255
    return result4


def contours(img):
    # img = cv2.GaussianBlur(img, (3, 3), 0)  # 用高斯平滑处理原图像降噪。若效果不好可调节高斯核大小
    # cv2.RETR_EXTERNAL:表示只检测外轮廓;cv2.CHAIN_APPROX_NONE:存储所有边界点
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 寻找轮廓
    mask = np.zeros(img.shape, np.uint8)
    cv2.drawContours(mask, contours, -1, (1, 0, 0), 1)  # 最后一个参数为轮廓的层数
    mask[0, :] = 0  # 去除边界轮廓
    mask[:, 0] = 0
    mask[img.shape[0] - 1, :] = 0
    mask[:, img.shape[1] - 1] = 0
    return mask


def canny(filename):
    #    binary1=imageio.imread(filename)
    binary1 = filename
    binary1 = binary1.astype(np.uint8)
    # canny提取轮廓
    binary = cv2.Canny(binary1, 255, 255 * 3, apertureSize=3)
    #    cv2.imwrite(savepath, binary, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
    return binary


def assign_spatial_reference_byfile(src_path, dst_path):
    src_ds = gdal.Open(src_path, gdal.GA_ReadOnly)
    sr = osr.SpatialReference()
    sr.ImportFromWkt(src_ds.GetProjectionRef())
    geoTransform = src_ds.GetGeoTransform()
    dst_ds = gdal.Open(dst_path, gdal.GA_Update)
    dst_ds.SetProjection(sr.ExportToWkt())
    dst_ds.SetGeoTransform(geoTransform)
    dst_ds = None
    src_ds = None


def makedirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(' Folder creation successful: {}'.format(dir_path))
    else:
        print(' Folder already exists: {}'.format(dir_path))
    return dir_path


# =========================================================================================================================
# 思路：
# 先给岸线图、分类图加上同样的坐标系。遍历岸线那条线上所有的像素点所在的位置，根据该位置去分类图上找到对应位置，
# 将分类图上该位置的值取出，专门存放在一个用于生成结果图的np数组中，遍历结束，输出该tif图
imgname = 'Type_of_Line1.tif'
savepath = makedirs(r"E:\model_data\test_new\test3\cut\cut_again")  # 保存路径
for k in range(1, 2):
    test = 'test' + str(k)
    srcpath = r"E:\model_data\test_new\test1\test1.tif"  # 地理信息源

    # 读取海陆分割结果图
    coastline_seg = imageio.imread(r"E:\model_data\test_new\test3\cut\cut_again\coastline_1.tif")
    coastline_seg[coastline_seg == 255] = 1
    # 岸线边缘提取
    line_result = contours(coastline_seg)

    # 读取分类图
    image_classify_result = imageio.imread(r"E:\model_data\test_new\test3\cut\cut_again\label_1.tif")
    # #读取岸线分类矩阵
    # 返回数组中所有coastline == 255的索引值,即提取所有岸线的索引值
    row, col = line_result.shape
    index = []
    ret = 0
    for j in range(col):
        for i in range(row):
            if line_result[i][j] == 1:
                index.append([i, j])
                ret += 1

    print(index)
    # shape[0]:岸线像素点总数
    for i in range(ret):  # 遍历岸线的所有像素点
        # 找出岸线每个像素的所在位置x,y
        # index里存的是每个岸线像素点的位置
        x = index[i][0]
        y = index[i][1]
        # y_predict2:分类图;x,y：坐标

        # 为什么6不行？

        GetListByCoord(image_classify_result,1 , x, y)  # 分类矩阵  半径为2 中心为xy
        # if (line_result[x][y] == 255):
        #     line_result[x][y] = 1

    cv2.imwrite(savepath + '\\' + imgname, line_result)
    assign_spatial_reference_byfile(srcpath, savepath + '\\' + imgname)  # coastline写入地理信息


