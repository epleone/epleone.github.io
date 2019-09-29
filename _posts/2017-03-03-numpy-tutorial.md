---
layout:     post
title:      "NumPy Tutorial"
subtitle:   "NumPy中文指南"
date:       2017-03-03 
author:     "epleone"
header-img: "img/post-bg-rwd.jpg"
tags:
    - NumPy
    - Python
---

## ndarray
NumPy的数组类被称作ndarray。在NumPy中维度(dimensions)叫做轴(axes)，轴的个数
叫做秩(rank)。

```
[[ 1., 0., 0.],
 [ 0., 1., 2.]]
```
在上面的例子中，数组的秩为2(它有两个维度).第一个维度长度为2,第二个维度长度为3.

ndarray对象的常用属性有：

- ***ndarray.ndim*** \
数组轴的个数，在python的世界中，轴的个数被称作秩

- **_ndarray.shape_** \
数组的维度。这是一个指示数组在每个维度上大小的整数元组。例如一个n排m列的矩阵，
它的shape属性将是(2,3),这个元组的长度显然是秩，即维度或者ndim属性

- **_ndarray.size_** \
数组元素的总个数，等于shape属性中元组元素的乘积。

- **_ndarray.dtype_** \
一个用来描述数组中元素类型的对象，可以通过创造或指定dtype使用标准Python类型。
另外NumPy提供它自己的数据类型。

- **_ndarray.itemsize_** \
数组中每个元素的字节大小。例如，一个元素类型为float64的数组itemsiz属性值为
8(=64/8),又如，一个元素类型为complex32的数组item属性为4(=32/8).

- **_ndarray.data_** \
包含实际数组元素的缓冲区，通常我们不需要使用这个属性，因为我们总是通过索引来
使用数组中的元素。

**Example :**

``` bat
>>> from numpy  import *
>>> a = arange(15).reshape(3, 5)
>>> a
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14]])
>>> a.shape
(3, 5)
>>> a.ndim
2
>>> a.dtype.name
'int32'
>>> a.itemsize
4
>>> a.size
15
>>> type(a)
numpy.ndarray
>>> b = array([6, 7, 8])
>>> b
array([6, 7, 8])
>>> type(b)
numpy.ndarray
```

## 创建数组
有好几种创建方式

``` bat
>>> c = array( [ [1,2], [3,4] ], dtype=complex )
>>> c
array([[ 1.+0.j,  2.+0.j],
       [ 3.+0.j,  4.+0.j]])
```

> 能用函数astype() 转换数据类型

- zeros 创建一个全是0的数组
- ones 创建一个全1的数组
- empty 创建一个内容随机并且依赖与内存状态的数组。

``` python
>>> zeros( (3,4) )
array([[0.,  0.,  0.,  0.],
       [0.,  0.,  0.,  0.],
       [0.,  0.,  0.,  0.]])
>>> ones( (2,3,4), dtype=int16 )                # dtype can also be specified
array([[[ 1, 1, 1, 1],
        [ 1, 1, 1, 1],
        [ 1, 1, 1, 1]],
       [[ 1, 1, 1, 1],
        [ 1, 1, 1, 1],
        [ 1, 1, 1, 1]]], dtype=int16)
>>> empty( (2,3) )
array([[  3.73603959e-262,   6.02658058e-154,   6.55490914e-260],
       [  5.30498948e-313,   3.14673309e-307,   1.00000000e+000]])
```
函数默认创建的数组类型(dtype)都是float64。

``` python
>>> np.linspace(2.0, 3.0, num=5)
array([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ])
>>> np.linspace(2.0, 3.0, num=5, endpoint=False)  # 不包括最后一个元素
array([ 2. ,  2.2,  2.4,  2.6,  2.8])
>>> np.linspace(2.0, 3.0, num=5, retstep=True)
(array([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ]), 0.25) # 返回间隔

>>> np.arange(3)
array([0, 1, 2])
>>> np.arange(3.0)
array([ 0.,  1.,  2.])
>>> np.arange(3,7)
array([3, 4, 5, 6])
>>> np.arange(3,7,2)
array([3, 5])
```

> 其它函数包括array, zeros, zeros_like, ones, ones_like,
 empty, empty_like, arange, linspace, rand, randn, fromfunction, fromfile


 ## 打印数组
当你打印一个数组，NumPy以类似嵌套列表的形式显示它，但是呈以下布局：

- 最后的轴从左到右打印
- 次后的轴从顶向下打印
- 剩下的轴从顶向下打印，每个切片通过一个空行与下一个隔开

一维数组被打印成行，二维数组成矩阵，三维数组成矩阵列表。

``` python
>>> a = arange(6)                         # 1d array
>>> print a
[0 1 2 3 4 5]
>>>
>>> b = arange(12).reshape(4,3)           # 2d array
>>> print b
[[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]]
>>>
>>> c = arange(24).reshape(2,3,4)         # 3d array
>>> print c
[[[ 0  1  2  3]
  [ 4  5  6  7]
  [ 8  9 10 11]]

 [[12 13 14 15]
  [16 17 18 19]
  [20 21 22 23]]]
```

如果一个数组用来打印太大了，NumPy自动省略中间部分而只打印角落, 
禁用NumPy的这种行为并强制打印整个数组，
你可以设置printoptions参数来更改打印选项。

``` python
>>> np.set_printoptions(threshold='nan')
```

## 基本运算
数组的算术运算是按元素的。新的数组被创建并且被结果填充。不像许多矩阵语言，
NumPy中的乘法运算符 * 指示按元素计算，矩阵乘法可以使用 dot 函数或创建矩阵对象
实现(参见教程中的矩阵章节)

``` python
>>> A = array( [[1,1],
...             [0,1]] )
>>> B = array( [[2,0],
...             [3,4]] )
>>> A*B                         # elementwise product
array([[2, 0],
       [0, 4]])
>>> dot(A,B)                    # matrix product
array([[5, 4],
       [3, 4]])
```


**有些操作符像 += 和 \*= 被用来更改已存在数组而不创建一个新的数组.**

许多非数组运算，如计算数组所有元素之和，被作为ndarray类的方法实现

``` python
>>> a = random.random((2,3))
>>> a
array([[ 0.6903007 ,  0.39168346,  0.16524769],
       [ 0.48819875,  0.77188505,  0.94792155]])
>>> a.sum()
3.4552372100521485
>>> a.min()
0.16524768654743593
>>> a.max()
0.9479215542670073
```

这些运算默认应用到数组好像它就是一个数字组成的列表，无关数组的形状。
你可以指定 axis 参数把运算应用到数组指定的轴上：

``` python
>>> b = arange(12).reshape(3,4)
>>> b
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>>
>>> b.sum(axis=0)                            # sum of each column
array([12, 15, 18, 21])
>>>
>>> b.min(axis=1)                            # min of each row
array([0, 4, 8])
>>>
>>> b.cumsum(axis=1)                         # cumulative sum along each row
array([[ 0,  1,  3,  6],
       [ 4,  9, 15, 22],
       [ 8, 17, 27, 38]])
```

## 通用函数(ufunc)

NumPy提供常见的数学函数如 sin , cos 和 exp 。
在NumPy中，这些叫作“通用函数”(ufunc)。
**在NumPy里这些函数作用按数组的元素运算，产生一个数组作为输出。** 


更多函数all, alltrue, any, apply along axis, argmax, argmin, argsort, average, 
bincount, ceil, clip, conj, conjugate, corrcoef, cov, cross, cumprod, cumsum, 
diff, dot, floor, inner, inv, lexsort, max, maximum, mean, median, min, minimum, 
nonzero, outer, prod, re, round, sometrue, sort, std, sum, trace, transpose, 
var, vdot, vectorize, where 
参见: [NumPy示例](https://docs.scipy.org/doc/numpy/reference/routines.html)


----

# 参考文献

1. [NumPy的详细教程]((http://blog.csdn.net/chen_shiqiang/article/details/51868115)).
2. [NumPy示例](https://docs.scipy.org/doc/numpy/reference/routines.html)
3. [Quickstart tutorial](https://docs.scipy.org/doc/numpy-dev/user/quickstart.html)