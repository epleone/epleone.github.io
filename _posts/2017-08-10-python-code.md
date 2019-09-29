---
layout:     post
title:      "记录一些Python的代码"
subtitle:   ""
date:       2017-08-10 
author:     "epleone"
header-img: "img/post-bg-rwd.jpg"
tags:
    - Python
    - NumPy
---

# openCV Code

---
### caffe前向过程
``` python
# test for MobileNet
import numpy as np
import cv2
import caffe

MODEL_FILE = './V2/mobilenet_deploy.prototxt'
PRETRAIN_FILE = r'./V2/MNext_iter_6000.caffemodel'
caffe.set_mode_cpu()
net = caffe.Net(MODEL_FILE, PRETRAIN_FILE, caffe.TEST)

mean_vec = np.array([103.94, 116.78, 123.68], dtype=np.float32)
reshaped_mean_vec = mean_vec.reshape(1, 1, 3)

im = cv2.imread(impath)
im = cv2.resize(im, (160, 120))
im = im[60-56:60+56, 80-56:80+56, ...]
im = im - reshaped_mean_vec
im *= 0.017
im = im.transpose((2, 0, 1))
im_ = np.zeros((1, 3, 112, 112), dtype=np.float32)
im_[0, ...] = im

starttime = datetime.datetime.now()
predictions = net.forward(data=im_)
endtime = datetime.datetime.now()
print((endtime.microsecond - starttime.microsecond)/1000, "ms")

pred = predictions['prob']
print(pred.argmax())
```
#NumPy Code
### 标量函数向量化

``` python
import numpy as np
from numpy import vectorize

def r_lookup(x):
    return x + 1
 
r_lookup_vec = vectorize(r_lookup)
a = np.arange(28*28).reshape(28, 28)
b = r_lookup_vec(a)
```

---


# 常用函数
``` python
# -*- coding: utf-8 -*-
'''
保存经常用到的python 函数
'''
import os
import shutil
import random


def list_op():
    list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    random.shuffle(list)   # 打乱列表
    slice = random.sample(list, 5)  # 从list中随机获取5个元素，作为一个片断返回
    return slice
    pass


def str2():
    # 字符串转为元组，返回：(1, 2, 3)
    print tuple(eval("(1,2,3)"))
    # 字符串转为列表，返回：[1, 2, 3]
    print list(eval("(1,2,3)"))
    # 字符串转为字典，返回：<type 'dict'>
    print type(eval("{'name':'ljq', 'age':24}"))


def genMatrix(rows, cols):
    matrix = [[0 for col in range(cols)] for row in range(rows)]
    return matrix


def PNG2JPG():
    from PIL import Image
    img = Image.open("D:\\1.png")
    img.save("1.jpeg", format="jpeg")


# 文件操作
def FileOP():
    # 目录操作：
    os.mkdir("file")
    # 创建目录
    # 复制文件：
    shutil.copyfile("oldfile", "newfile")  # oldfile和newfile都只能是文件
    shutil.copy("oldfile", "newfile")   # oldfile只能是文件夹，newfile可以是文件，也可以是目标目录

    # 复制文件夹：
    shutil.copytree("olddir", "newdir")  # olddir和newdir都只能是目录，且newdir必须不存在
    # 重命名文件（目录）
    os.rename("oldname", "newname")
    # 文件或目录都是使用这条命令
    # 移动文件（目录）
    shutil.move("oldpos", "newpos")
    # 删除文件
    os.remove("file")
    # 删除目录
    os.rmdir("dir")
    # 只能删除空目录
    shutil.rmtree("dir")
    # 空目录、有内容的目录都可以删
    # 转换目录
    os.chdir("path")
    # 换路径


def FunOP():
    l = [1, 3, 5, 6, 7, 8]
    b = map(lambda a, b: [a, b], l[::2], l[1::2])  # b = [[1,3],[5,6],[7,8]]
    return b
    pass


def getAllLabelTxt(root_path):
    os.system('dir ' + root_path + '\\*.txt /b/s> file_list.txt')
    txt = open('file_list.txt', 'r').readlines()
    # os.remove('file_list.txt')
    return txt

	
def saveList(man):
	try:
		man_file=open('man.txt', 'w')      # 以w模式访问文件man.txt
		other_file=open('other.txt','w')   # 以w模式访问文件other.txt
		print (man, file=man_file)         # 将列表man的内容写到文件中
		print (other, file=other_file)
	except IOError:
		print ('File error')
	finally:
		man_file.close()
		other_file.close()

```


### python版本控制

``` python
import sys
if sys.version_info < (3, 4):
    raise RuntimeError('At least Python 3.4 is required')
```
