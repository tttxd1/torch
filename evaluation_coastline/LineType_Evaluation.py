import cv2,imageio
from skimage import measure,data,color,io,morphology
from osgeo import gdal,osr
import tkinter as tk
import numpy as np
import os,glob
from itertools import chain

def makedirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(' Folder creation successful: {}'.format(dir_path))
    else:
        print(' Folder already exists: {}'.format(dir_path))
    return dir_path


def InitList(x, y):
    array = [([0] * y) for i in range(x)]
    return array

# 预测图，真值图
# 真值图，预测图
def GetListByCoord(array, array2, radius, x, y):
    # 根据半径来确定数组的行数和列数
    row_col = 2 * radius + 1
    # 初始化结果数组
    result = InitList(row_col, row_col)
    # 获取传入的array的行数和列数
    arrayRow, arrayCol = len(array), len(array[0])
    # 坐标x、y的值即为结果数组的中心，依此为偏移
    for i in range(result.__len__()):
        for j in range(result.__len__()):
            # 是否越界
            # print(i)
            if (i + x - radius < 0 or j + y - radius < 0 or i + x - radius >= arrayRow or j + y - radius >= arrayCol):
                continue
                # 预测为正例，并且标签真是正例
                # 标签为正例，并且预测也为正例
            elif (array[x, y] == array2[i + x - radius, j + y - radius]):
                return 1
    return 0

predict_image = io.imread(r"E:\model_data\Ablation_Study\Ablation_Again\result\FarSeg_Super_NoSLE\Type_of_Line1.tif")  # y预测图
true = io.imread(r"E:\model_data\test_new\test2\Type_of_Line2.tif")  # 真值图
# # 比重数组
# proportion = []
# # 各类别数
# num_set = []

savepath = r"E:\model_data\Ablation_Study\Ablation_Again\result\FarSeg_Super_NoSLE"
# distpath = makedirs(os.path.join(savepath, name))  # 保存路径

data = open(os.path.join(savepath, 'test2_2' + '.txt'), "w", encoding='utf-8')
unique = np.delete(np.unique(true), 0)
# 总点数
k = np.argwhere(true != 0)
sum = len(k)
proportion_P = 0
proportion_R = 0
proportion_F = 0
for k in range(len(unique)):
    max = 0
    BP = 0
    BR = 0
    F1 = 0
    # 找到第k类在预测图中的所有位置
    index_predict = np.argwhere(predict_image == unique[k])
    # 每一类的数量
    num = len(index_predict)
    # 找到第k类在真值图中的所有位置
    index_true = np.argwhere(true == unique[k])

    k_pre_num = 0
    k_true_num = 0
    # 精确率
    for i in range(index_predict.shape[0]):
        x = index_predict[i][0]
        y = index_predict[i][1] # 预测结果是true
        # 接下来要去找真值图中相应位置也是true的点的数量
        k_pre_num += GetListByCoord(predict_image, true, 1, x, y)  # 分类矩阵  半径为2 中心为xy

    # 召回率
    for ii in range(index_true.shape[0]):
        x = index_true[ii][0]
        y = index_true[ii][1]
        k_true_num += GetListByCoord(true, predict_image, 1, x, y)

    if index_predict.shape[0] and k_pre_num and k_true_num:
        BP = k_pre_num / index_predict.shape[0]
        # print(BP)
        BR = k_true_num / index_true.shape[0]
        # print(BR)
        F1 = (2 * BP * BR) / (BR + BP)
    proportion = num / sum
    proportion_P += proportion * BP
    proportion_R += proportion * BR
    proportion_F += proportion * F1

    print("{}  BP = {:.4f} BR={:.4f}  F1 = {:.4f}".format(unique[k], BP, BR, F1), file=data)
    print("F1={:.6f},预测正确像素={},预测总像素={} 真值召回={}  真值总像素={}".format(F1, k_pre_num, index_predict.shape[0], k_true_num, index_true.shape[0]))

print(proportion_P)
print(proportion_R)
print(proportion_F)
# data1 = open(savepath + '\line_data1.txt', 'a+')
print("BP_all = {:.4f} BR_all={:.4f}  F1_all = {:.4f}".format(proportion_P, proportion_R, proportion_F), file=data)
data.close()