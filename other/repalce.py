import os


# a = ['abc']
# a = a[0].replace('abc', 'ss')
#
# print(a)

path = r"C:\Users\224\Downloads\极市打榜新手任务_编码开发(二).784461660.ai-zh.txt"
save_path = r"C:\Users\224\Downloads\极市打榜新手任务_编码开发(二).txt"
with open(path, "r",encoding='utf-8') as f:  # 打开文件
    data = f.readlines()  # 读取文件
# print(data[1])
for i in range(len(data)):
    print(data[i])
    for j in range(10):
        data[i] = data[i].replace(str(j), '')
    data[i] = data[i].replace('<font color="#FFFFFF" size="">', '')
    data[i] = data[i].replace('</font>', '')
    data[i] = data[i].replace(':', '')
    data[i] = data[i].replace(', --> ,', '')
    print(data[i])

with open(save_path, "w", encoding='utf-8') as f:  # 打开文件
    for i in range(len(data)):
        f.write(data[i])