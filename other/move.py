import shutil
from glob import glob
from pathlib import Path
import os


# srcfile 需要复制、移动的文件
# dstpath 目的地址

path = glob(r"G:\数据集\连云港\NewDataSet_5.18\pre_image_train\*.tif")

class_path = r"G:\数据集\连云港\NewDataSet_5.18\coastline_classify_label_train"
coastline_path =r"G:\数据集\连云港\NewDataSet_5.18\coastline_train"
line_path = r"G:\数据集\连云港\NewDataSet_5.18\line_train"

new_image_path = r"E:\yqj\try\code\torch\Train\Data\coastline\image\train"
new_class_path = r"E:\yqj\try\code\torch\Train\Data\coastline\label\coastline_classify_label_train"
new_coastline_path = r"E:\yqj\try\code\torch\Train\Data\coastline\label\coastline_train"
new_line_path = r"E:\yqj\try\code\torch\Train\Data\coastline\label\line_train"


for i in range(len(path)):
    srcfile = path[i]
    fpath, fname = os.path.split(srcfile)

    class_file = class_path + "\\" + fname
    coastline_file = coastline_path + "\\" + fname
    line_file = line_path + "\\" + fname

    shutil.copy(srcfile, new_image_path + '\\'+ str(i + 1) + ".tif")  # 复制文件
    shutil.copy(class_file, new_class_path  + '\\' + str(i + 1) + ".tif")  # 复制文件
    shutil.copy(coastline_file, new_coastline_path + '\\'  + str(i + 1) + ".tif")  # 复制文件
    shutil.copy(line_file, new_line_path + '\\' + str(i + 1) + ".tif")  # 复制文件
print("end")


