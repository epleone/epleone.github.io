---
layout:     post
title:      "Tips for Caffe/Ubuntu"
subtitle:   "Caffe/Ubuntu的一些小技巧"
date:       2017-09-06
author:     "epleone"
header-img: "img/post-bg-android.jpg"
tags:
    - Caffe
    - Ubuntu
    - CUDA
    - Tips
---


# Ubuntu下清屏等终端常用命令

| 快捷键          | 功能         |
| ------------ | ---------- |
| **ctrl + l** | 清屏         |
| **ctrl + c** | 终止命令       |
| **ctrl + u** | 清除光标到行首的字符 |
| **ctrl + k** | 清除光标到行尾的字符 |
| **ctrl + w** | 清除光标之前一个单词 |


# nvcc warning : The 'compute_20', 'sm_20', and 'sm_21' architectures are deprecated

安装的CUDA8.0版本之后，在编译caffe 的Mkefile.config 时会遇到报错

> nvcc warning : The 'compute_20', 'sm_20', and 'sm_21' architectures are deprecated, and may be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).

原因是Makefile中采用了CUDA的compute capability 2.0和2.1，但是这两种计算能力从CUDA 8.0开始被弃用了，
所以可以将**-gencode arch=compute_20,code=sm_20** 和**-gencode arch=compute_20,code=sm_21**这两行删除即可。

# caffe makefile.config
按照[这段折腾 caffe 的日子](http://blog.csdn.net/u010167269/article/details/50703948 'Title') 配置好caffe环境之后
hdf5的路径不对，需要把他加到文件里面来
Makefile.config 更改如下
``` 
92 # Whatever else you find you need goes here.
93 INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial/
94 LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu/hdf5/serial
```

# cuda链接报错
```
# 这个是8.0的 根据cuda的版本做相应的改动
sudo cp /usr/local/cuda-8.0/lib64/libcudart.so.8.0 /usr/local/lib/libcudart.so.8.0 && sudo ldconfig 
sudo cp /usr/local/cuda-8.0/lib64/libcublas.so.8.0 /usr/local/lib/libcublas.so.8.0 && sudo ldconfig 
sudo cp /usr/local/cuda-8.0/lib64/libcurand.so.8.0 /usr/local/lib/libcurand.so.8.0 && sudo ldconfig
```

# ubuntu添加新用户
``` bash
# add
sudo useradd -r -m -s /bin/bash #username#
sudo passwd #username#

# del
sudo userdel #username#

# 如果报错 'user is currently used by process'
ps -u #username# | awk '{print $1}' | grep -vi pid | xargs kill -9 && deluser #username#
```

# 普通用户提升到root权限

1、编辑passwd文件

```
sudo vim /etc/passwd
```

2、找到你想提权的用户（比如test），将用户名后面的数字改成0
找到用户test
test:x:999:999::/home/test

修改权限
test:x:0:0::/home/test

方法二、临时使用root用户
```
su root
```

# Linux注销用户
```
首先用who命令或w命令，列出登陆的用户列表
pkill -kill -t pts/# (#表示要注销的pts号)
```

# 登陆或者切换到其他用户
```
su - #username#
```

# ssh 后台输出
```
nohup ./XX.sh &

#不输出日志
nohup ./XX.sh >/dev/null 2>&1
```
