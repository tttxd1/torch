import os
import sys
from PIL import Image

import numpy as np
import json
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score


"""
本文件主要实现以下功能：
1、评估模型，包括准确率、精准率、召回率、F1分数
"""

# ========================================================================================================================================

def calculation(y_label, y_pre, row, col):
    '''
    本函数主要计算以下评估标准的值：
    1、精准率
    2、召回率
    3、F1分数
    '''

    # 转成列向量
    y_label = np.reshape(y_label, (row * col, 1))
    y_pre = np.reshape(y_pre, (row * col, 1))

    y_label.astype('float64')
    y_pre.astype('float64')

    # 精准率
    precision = precision_score(y_label, y_pre, average=None)

    # 召回率
    recall = recall_score(y_label, y_pre, average=None)

    # F1
    f1 = f1_score(y_label, y_pre, average=None)

    # kappa
    kappa = cohen_kappa_score(y_label, y_pre)

    return precision, recall, f1, kappa


# ========================================================================================================================================

def estimate(y_label, y_pred, dirname):


    # 准确率
    acc = np.mean(np.equal(y_label, y_pred) + 0)

    precision, recall, f1, kappa = calculation(y_label, y_pred, y_label.shape[0], y_label.shape[1])

    # print(precision)
    # print(recall)

    precision = round(np.mean(precision), 5)
    recall = round( np.mean(recall), 5)
    f1 = round(np.mean(f1), 5)
    kappa = round(kappa, 5)
    acc = round(acc, 5)


    te = []

    te.append(acc)
    te.append(precision)
    te.append(recall)
    te.append(f1)
    te.append(kappa)


    dirPath = os.path.join(dirname, 'Acc.txt')
    if os.path.exists(dirPath):
        os.remove(dirPath)

    with open(os.path.join(dirname, 'Acc.txt'), 'a', encoding="utf-8") as f:
        f.write(json.dumps(te, ensure_ascii=False))


if __name__ == '__main__':
    # 读取真值图和预测图


    #
    # imgPath = sys.argv[1]
    # labelPath = sys.argv[2]


    imgPath = r"E:\yqj\测试\image\珊瑚礁\羚羊礁结果图.tif"
    labelPath = r"E:\yqj\测试\image\珊瑚礁\lingyangjiao_lable.tif"

    str = imgPath.split('\\')
    dirname = imgPath[:-(len(str[-1]) + 1)]


    true_img = Image.open(imgPath)

    true_img = np.array(true_img)
    print(true_img.shape)


    pred_img = Image.open(labelPath)
    pred_img = np.array(pred_img)
    estimate(true_img, pred_img, dirname)





