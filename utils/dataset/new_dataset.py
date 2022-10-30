import glob
import os

# imagePath = os.path.join(os.getcwd(), 'Train', "shanhujiao", "image")
# labelPath = os.path.join(os.getcwd(), 'Train', "shanhujiao", "label")

imagePath = r"G:\数据集\YQJ\cut\train\image"
labelPath = r"G:\数据集\YQJ\cut\train\label"

# print(os.listdir(imagePath))
# print(os.listdir(labelPath))

images_path_list = glob.glob(os.path.join(imagePath, '*.tif'))

print(images_path_list)
images_path_list.sort()
print(images_path_list)
# path = os.listdir(imagePath)
# path.sort()
# print(path)
#
# name = path[0].split('\\') #图片名称
# number = name[-1].replace('.tif', '').split('_')  #数字标号  [0]:第几张图片 [1]:行数 [2]:列数
# print(number)


