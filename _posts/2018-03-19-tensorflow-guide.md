---
layout:     post
title:      "TensorFlow Guide"
subtitle:   "TensorFlow的心得体会"
date:       2018-03-19
author:     "epleone"
header-img: "img/post-bg-e2e-ux.jpg"
tags:
    - TensorFlow
    - Tips
---

# TensorFlow 学习心得

[TOC]

## 前言

TensorFlow 是一个编程系统, 使用图来表示计算任务. 图中的节点被称之为 ops (operation 的缩写). 一个 op 获得 0 个或多个 `Tensor`, 执行计算, 产生 0 个或多个 `Tensor`. 每个 Tensor 是一个类型化的多维数组. 例如, 你可以将一小组图像集表示为一个四维浮点数数组, 这四个维度分别是 `[batch, height, width, channels]`.

一个 TensorFlow 的`graph` 描述了计算的过程. 为了进行计算, graph必须在 `Session ` 里被启动. Session将`graph`的 op 分发到诸如 CPU 或 GPU 之类的 `device` 上, 同时提供执行 op 的方法. 这些方法执行后, 将产生的 tensor 返回. 在 Python 语言中, 返回的 tensor 是 [numpy](http://www.numpy.org/) `ndarray` 对象; 在 C 和 C++ 语言中, 返回的 tensor 是 `tensorflow::Tensor` 实例.

TensorFlow 程序一般可划分为两个流程：

- construction phase(构建过程)，会构建出一个图（graph），即所谓的计算图（computation graph）
- evaluation phase(执行过程)，使用 session 执行构建过程中生成的图中的操作




这篇文章是我根据网上的blog以及分享整理而成，加了一些自己的心得总结，但不是所有的代码都验证过，可能有些代码存在错误。

我是用的tensorflow版本是1.4.1，GPU版本。

可以通过下面的代码获取:

``` shell
# ubuntu
sudo pip install tensorflow-gpu==1.4.1

# windows
pip install tensorflow==1.4.0
```



验证是否安装成功。

``` shell
> python
> import tensorflow as tf
> 
```






## Session

`tf.Session.__init__(target=”, graph=None, config=None)`  

Session()构造方法有3个可选参数。target指定执行引擎，默认空字符串，分布式中用于连接不同tf.train.Server实例。graph加载Graph对象，默认None，默认当前数据流图，区分多个数据流图时的执行，不在with语句块内创建Session对象。config指定Session对象配置选项,比如设备信息，这部分可以参考 <a href="#Device">本文中的设备那一节</a> 。

> **target:**（可选）连接的执行引擎，默认是使用in-process引擎，分布式TensorFLow有更多的例子。 
> **graph:** (可选)投放进的计算图（graph），要是没有指定的话，那么默认的图就会被投放到这个session。**要是你在同一个进程里面用了很多的图，你将为各个图使用不用的session，但是每一个graph都能够在多个session中使用。**在这种情况下，经常显式的传递graph参数到session的构造里面。 
> **config:** (可选) A ConfigProto protocol buffer with configuration options for the session.

### tf.session() 

tf.Session():需要在启动session之前构建整个计算图，然后启动该计算图。

使用tf.Session()来构建会话我们需要在会话构建之前定义好全部的操作（operation）然后再构建会话。

### tf.InteractiveSession()

tf.InteractiveSession():它能让你在运行图的时候，插入一些计算图，这些计算图是由某些操作(operations)构成的。我们在使用tf.InteractiveSession()来构建会话的时候，我们可以先构建一个session然后再定义操作（operation)。这对于工作在交互式环境中的人们来说非常便利，比如使用IPython。==尽量不要用==。

### sess.run()和Tensor.eval()

c.eval() 等价于 sess.run(c)，是其语法糖形式。

tf.Tensor.eval() 其实就是Session.run() 的另外一种写法：

```python
with tf.Session() as sess:
  print(accuracy.eval({x:mnist.test.images,y_: mnist.test.labels}))
```

上面代码的效果和下面的代码是等价的：

```python
with tf.Session() as sess:
  print(sess.run(accuracy, {x:mnist.test.images,y_: mnist.test.labels}))
```

但是要注意的是，eval()只能用于tf.Tensor类对象，也就是有输出的Operation。对于没有输出的Operation, 可以用.run()或者Session.run()。Session.run()没有这个限制。





## Graph 

Graph(图)是tensorflow的核心，所有的操作都是基于图进行的，图中有很多的op，一个op又有一个或则多个的Tensor构成。一个Session里面可以包含多个图。同时一张图也可以在多个Session中运行。

### tf.get_default_graph()

当我们导入`tensorflow`包的时候，系统已经帮助我们产生了一个默认的图，它被存在`_default_graph_stack`中，但是我们没有权限直接进入这个图，我们需要使用`tf.get_default_graph()`命令来获取图。

```python
graph = tf.get_default_graph()
```

`tensorflow`中的图上的节点被称之为`operations`或者`ops`。我们可以使用`graph.get_operations()`命令来获取图中的`operations`。

- get_operations()：

  ```
  # graph = tf.Graph() # 空白图
  graph = tf.get_default_graph()
  names = [op.name for op in model.graph.get_operations() if op.type=='Conv2D']12
  ```

- tensor = tf.get_default_graph().get_tensor_by_name(tensor_name)

  - Get the tensor with this name.

### Graph.as_default()

另外一种典型的用法就是要使用到`Graph.as_default()` 的上下文管理器（ context manager），它能够在这个上下文里面覆盖默认的图。如下例:

~~~ python
import tensorflow as tf
import numpy as np

c=tf.constant(value=1)
#print(assert c.graph is tf.get_default_graph())
print(c.graph)
print(tf.get_default_graph())

g=tf.Graph()
print("g:",g)
with g.as_default():
    d=tf.constant(value=2)
    print(d.graph)
    #print(g)

g2=tf.Graph()
print("g2:",g2)
g2.as_default()
e=tf.constant(value=15)
print(e.graph)
~~~

上面的例子里面创创建了一个新的图g，然后把g设为默认，那么接下来的操作不是在默认的图中，而是在g中了。你也可以认为现在g这个图就是新的默认的图了。 
要注意的是，最后一个量e不是定义在with语句里面的，也就是说，e会包含在最开始的那个图中。也就是说，要在某个graph里面定义量，要在with语句的范围里面定义。

一个Graph的实例支持任意多数量通过名字区分的的“collections”。 
为了方便，当构建一个大的图的时候，collection能够存储很多类似的对象。比如 tf.Variable就使用了一个collection（tf.GraphKeys.GLOBAL_VARIABLES），包含了所有在图创建过程中的变量。 
也可以通过之指定新名称定义其他的collection

参考链接：

1. [小白学Tensorflow之可视化与图](https://www.jianshu.com/p/5080d45d39da) 
2. [TensorFlow学习（三）：Graph和Session](http://blog.csdn.net/xierhacker/article/details/53860379) 





## Saver

`tf.train.Saver.init` 

```python
__init__(
    var_list=None,
    reshape=False,
    sharded=False,
    max_to_keep=5,
    keep_checkpoint_every_n_hours=10000.0,
    name=None,
    restore_sequentially=False,
    saver_def=None,
    builder=None,
    defer_build=False,
    allow_empty=False,
    write_version=tf.train.SaverDef.V2,
    pad_step_number=False,
    save_relative_paths=False,
    filename=None
)
```

其中比较重要的参数有:

-   `max_to_keep` : 指示要保留的最近检查点文件的最大数量。当新文件被创建时，旧文件被删除。如果是 None或0，则保留所有检查点文件。默认为5（也就是保留5个最近的检查点文件。）
-   `keep_checkpoint_every_n_hours` ：除了保留最新的max_to_keep检查点文件之外，您可能希望每N个小时的训练都保留一个检查点文件。如果您想稍后分析长时间训练期间模型的进展情况，这可能很有用。例如，传递 keep_checkpoint_every_n_hours = 2可确保每2小时的训练保留一个检查点文件。 默认值为10,000小时
-   `var_list` : 指定将被保存和恢复的变量。示例可以参考 <a href="#权值保存">权值保存</a> 那一节。它可以作为字典或列表传递：
    -   A `dict` of names to variables: The keys are the names that will be used to save or restore the variables in the checkpoint files.
    -   A list of variables: The variables will be keyed with their op name in the checkpoint files.
-   `reshape` : if `True`, 允许从变量具有不同形状但具有相同数量的元素和类型的保存文件恢复变量。 如果您重新设计了变量并希望从旧checkpoint重新加载该变量，这非常有用。



Graph存储模型结构，而Saver则是存储和恢复权值变量的具体数值。由于TensorFlow 的版本一直在更新, 保存模型的方法也发生了改变。在python 环境,和在C++ 环境(移动端) 等不同的平台需要的模型文件也是不也一样的。

参考下面的资料做一个详细的介绍:    [1.](https://stackoverflow.com/questions/44516609/tensorflow-what-is-the-relationship-between-ckpt-file-and-ckpt-meta-and-ckp)   [2.](https://stackoverflow.com/questions/35508866/tensorflow-different-ways-to-export-and-run-graph-in-c/43639305#43639305) 

最准确的请参考Tensorflow官方文档:   [3.](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/session_bundle/README.md])   [4.](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md) 

这里主要介绍checkpoint。模型保存后产生四个文件，分别是：

~~~ bash
|--models
|    |--checkpoint
|    |--.meta
|    |--.data
|    |--.index
~~~

其中checkpoint文件是个文本文件，里面记录了保存的最新的checkpoint文件以及其它checkpoint文件列表，

.meta文件保存了当前图结构

.index文件保存了当前参数名

.data文件保存了当前参数值.

~~~ python
ckpt = tf.train.get_checkpoint_state(./models/)
print(ckpt.model_checkpoint_path)
~~~

`tf.train.import_meta_graph` 函数给出.meta路径后会加载图结构，返回Saver对象

`tf.train.Saver` 函数则返回加载默认图的saver对象。[参考这篇blog](https://www.cnblogs.com/hellcat/p/6925757.html)  

~~~ python
# 连同图结构一同加载
ckpt = tf.train.get_checkpoint_state('./model/')
saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path +'.meta')
with tf.Session() as sess:
    saver.restore(sess,ckpt.model_checkpoint_path)
             
# 只加载数据，不加载图结构，可以在新图中改变batch_size等的值
# 不过需要注意，Saver对象实例化之前需要定义好新的图结构，否则会报错
saver = tf.train.Saver()
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('./model/')
    saver.restore(sess,ckpt.model_checkpoint_path)
~~~



生成图协议文件(graph proto file)，二进制文件，扩展名.pb，tf.tran.write_graph()保存，只包含图形结构，不包含权重，tf.import_graph_def加载图形。[参考这里](https://zhuanlan.zhihu.com/p/28710966)



### 权值保存  

`saver.save(session,dir[,global_step])` 

参数说明:

-   `session` : 执行的session
-   `dir` : 保存的目录
-   `global_step` : 设置为文件名编号, 一般配合 if step % 1000 == 0 使用

```python
saver = tf.train.Saver()
saver.save(sess, 'my-model', global_step=0) ==> filename: 'my-model-0'
saver.save(sess, 'my-model', global_step=1000) ==> filename: 'my-model-1000'
```



### 权值加载

`saver.restore(sess, checkpoint_path)` 

参数说明:

-   `session` : 执行的session
-   `checkpoint_path` :  checkpoint所在的目录路径

有时候，我们只保存和加载模型的部分参数。例如，你已经训练了一个5层的神经网络；现在你想训练一个新的神经网络，它有6层。加载旧模型的参数作为新神经网络前5层的参数。 
通过传递给tf.train.Saver()一个Python字典，你可以简单地指定名字和想要保存的变量。字典的keys是保存在磁盘上的名字，values是变量的值。

~~~ python
# 创建一些对象
v1 = tf.Variable(..., name="v1")
v2 = tf.Variable(..., name="v2")
...
# 添加一个节点用于保存和加载变量v2，使用名字“my_v2”
saver = tf.train.Saver({"my_v2": v2})
# 然后，加载模型，使用saver对象从磁盘上加载变量，之后再使用模型进行一些操作
with tf.Session() as sess:
  # 从磁盘上加载对象
  saver.restore(sess, "/tmp/model.ckpt")
~~~

注意一下两点：

- 如果需要保存和恢复模型变量的不同子集，可以创建任意多个saver对象。同一个变量可被列入多个saver对象中，只有当saver的`restore()`函数被运行时，它的值才会发生改变。
- 如果你仅在session开始时恢复模型变量的一个子集，你需要对剩下的变量执行初始化op。详情请见[`tf.initialize_variables()`](https://github.com/jikexueyuanwiki/tensorflow-zh/blob/master/api_docs/python/state_ops.md)。



### 图存储加载

仅保存图模型，图写入二进制协议文件：

``` python
v = tf.Variable(0,name='my_variable')
with tf.Session() as sess:
    tf.train.write_graph(sess.graph_def,'/tmp/tfmodel','train.pbtxt')

```

​    

图模型读取:

``` python
 with tf.Session() as _sess:
        with grile.FastGFile("/tmp/tfmodel/train.pbtxt",'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _sess.graph.as_default()
            tf.import_graph_def(graph_def,name='tfgraph')

```

  

### .pb 模型

谷歌推荐的保存模型的方式是保存模型为 ==PB 文件== ，它具有语言独立性，可独立运行，封闭的序列化格式，任何语言都可以解析它，它允许其他语言和深度学习框架读取、继续训练和迁移 TensorFlow 的模型。它的主要使用场景是实现**创建模型与使用模型的解耦， 使得前向推导 inference的代码统一。另外的好处是保存为 PB 文件时候，模型的变量都会变成固定的，导致模型的大小会大大减小，适合在手机端运行。

这种 PB 文件是表示 MetaGraph 的 protocol buffer格式的文件，**MetaGraph 包括计算图，数据流，以及相关的变量和输入输出**signature以及 asserts 指创建计算图时额外的文件。



生成模型的 PB 文件。

```
import tensorflow as tf
import os
from tensorflow.python.framework import graph_util

pb_file_path = os.getcwd()

with tf.Session(graph=tf.Graph()) as sess:
    x = tf.placeholder(tf.int32, name='x')
    y = tf.placeholder(tf.int32, name='y')
    b = tf.Variable(1, name='b')
    xy = tf.multiply(x, y)
    # 这里的输出需要加上name属性
    op = tf.add(xy, b, name='op_to_store')
    sess.run(tf.global_variables_initializer())
    # convert_variables_to_constants 需要指定output_node_names，list()，可以多个
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['op_to_store'])

    # 测试 OP
    feed_dict = {x: 10, y: 3}
    print(sess.run(op, feed_dict))

    # 写入序列化的 PB 文件
    with tf.gfile.FastGFile(pb_file_path+'model.pb', mode='wb') as f:
        f.write(constant_graph.SerializeToString())

    # INFO:tensorflow:Froze 1 variables.
    # Converted 1 variables to const ops.
    
    # 官网有误，写成了 saved_model_builder  
    builder = tf.saved_model.builder.SavedModelBuilder(pb_file_path+'savemodel')
    # 构造模型保存的内容，指定要保存的 session，特定的 tag, 
    # 输入输出信息字典，额外的信息
    builder.add_meta_graph_and_variables(sess,
                                       ['cpu_server_1'])


# 添加第二个 MetaGraphDef 
#with tf.Session(graph=tf.Graph()) as sess:
#  ...
#  builder.add_meta_graph([tag_constants.SERVING])
#...

builder.save()  # 保存 PB 模型
```

保存好以后到saved_model_dir目录下，会有一个saved_model.pb文件以及variables文件夹。顾名思义，variables保存所有变量，saved_model.pb用于保存模型结构等信息。

这种方法对应的导入模型的方法：

```
with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, ['cpu_1'], pb_file_path+'savemodel')
    sess.run(tf.global_variables_initializer())

    input_x = sess.graph.get_tensor_by_name('x:0')
    input_y = sess.graph.get_tensor_by_name('y:0')

    op = sess.graph.get_tensor_by_name('op_to_store:0')

    ret = sess.run(op,  feed_dict={input_x: 5, input_y: 5})
    print(ret)
# 只需要指定要恢复模型的 session，模型的 tag，模型的保存路径即可,使用起来更加简单
```

这样和之前的导入 PB 模型一样，也是要知道tensor的name。那么如何可以在不知道tensor name的情况下使用呢，实现彻底的解耦呢？ 给`add_meta_graph_and_variables`方法传入第三个参数，`signature_def_map`即可。



参考链接：

1. [学习笔记TF049:TensorFlow 模型存储加载...](https://zhuanlan.zhihu.com/p/28710966)
2. [变量:创建、初始化、保存和加载](http://wiki.jikexueyuan.com/project/tensorflow-zh/how_tos/variables.html) 
3. [TensorFlow学习系列（三）：保存/恢复和混合多个模型](https://www.jianshu.com/p/8487db911d9a) 
4. [[『TensorFlow』模型载入方法汇总](http://www.cnblogs.com/hellcat/p/6925757.html)](https://www.cnblogs.com/hellcat/p/6925757.html) 
5. [TensorFlow 保存模型为 PB 文件](https://zhuanlan.zhihu.com/p/32887066) 





## Device

在一套标准的系统上通常有多个计算设备. TensorFlow 支持 CPU 和 GPU 这两种设备. 我们用指定字符串 `strings` 来标识这些设备. 比如:

-   `"/cpu:0"`: 机器中的 CPU
-   `"/gpu:0"`: 机器中的 GPU, 如果你有一个的话.
-   `"/gpu:1"`: 机器中的第二个 GPU, 以此类推...

如果一个 TensorFlow 的 operation 中兼有 CPU 和 GPU 的实现, 当这个算子被指派设备时, GPU 有优先权. 比如`matmul`中 CPU 和 GPU kernel 函数都存在. 那么在 `cpu:0` 和 `gpu:0` 中, `matmul` operation 会被指派给 `gpu:0` .



### 记录设备指派情况

为了获取你的 operations 和 Tensor 被指派到哪个设备上运行, 用 `log_device_placement` 新建一个 `session`, 并设置为 `True`.

```
# 新建一个 graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# 新建session with log_device_placement并设置为True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# 运行这个 op.
print sess.run(c)
```

你应该能看见以下输出:

```
Device mapping:
/job:localhost/replica:0/task:0/gpu:0 -> device: 0, name: Tesla K40c, pci bus
id: 0000:05:00.0
b: /job:localhost/replica:0/task:0/gpu:0
a: /job:localhost/replica:0/task:0/gpu:0
MatMul: /job:localhost/replica:0/task:0/gpu:0
[[ 22.  28.]
 [ 49.  64.]]
```



### 手工指派设备

如果你不想使用系统来为 operation 指派设备, 而是手工指派设备, 你可以用 `with tf.device` 创建一个设备环境, 这个环境下的 operation 都统一运行在环境指定的设备上.

```python
# 新建一个graph.
with tf.device('/cpu:0'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# 新建session with log_device_placement并设置为True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# 运行这个op.
print sess.run(c)
```

你会发现现在 `a` 和 `b` 操作都被指派给了 `cpu:0`.

```cmd
Device mapping:
/job:localhost/replica:0/task:0/gpu:0 -> device: 0, name: Tesla K40c, pci bus
id: 0000:05:00.0
b: /job:localhost/replica:0/task:0/cpu:0
a: /job:localhost/replica:0/task:0/cpu:0
MatMul: /job:localhost/replica:0/task:0/gpu:0
[[ 22.  28.]
 [ 49.  64.]]
```



###在多GPU系统里使用单一GPU

如果你的系统里有多个 GPU, 那么 ID 最小的 GPU 会默认使用. 如果你想用别的 GPU, 可以用下面的方法显式的声明你的偏好:

```python
# 新建一个 graph.
with tf.device('/gpu:2'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
  c = tf.matmul(a, b)
# 新建 session with log_device_placement 并设置为 True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# 运行这个 op.
print sess.run(c)
```

如果你指定的设备不存在, 你会收到 `InvalidArgumentError` 错误提示:

```CMD
InvalidArgumentError: Invalid argument: Cannot assign a device to node 'b':
Could not satisfy explicit device specification '/gpu:2'
   [[Node: b = Const[dtype=DT_FLOAT, value=Tensor<type: float shape: [3,2]
   values: 1 2 3...>, _device="/gpu:2"]()]]
```

为了避免出现你指定的设备不存在这种情况, 你可以在创建的 `session` 里把参数 `allow_soft_placement` 设置为 `True`, 这样 tensorFlow 会自动选择一个存在并且支持的设备来运行 operation.

```python
# 新建一个 graph.
with tf.device('/gpu:2'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
  c = tf.matmul(a, b)
# 新建 session with log_device_placement 并设置为 True.
sess = tf.Session(config=tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=True))
# 运行这个 op.
print sess.run(c)
```



###使用多个 GPU

如果你想让 TensorFlow 在多个 GPU 上运行, 你可以建立 multi-tower 结构, 在这个结构 里每个 tower 分别被指配给不同的 GPU 运行. 比如:

```python
# 新建一个 graph.
c = []
for d in ['/gpu:2', '/gpu:3']:
  with tf.device(d):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
    c.append(tf.matmul(a, b))
with tf.device('/cpu:0'):
  sum = tf.add_n(c)
# 新建session with log_device_placement并设置为True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# 运行这个op.
print sess.run(sum)
```

你会看到如下输出:

```cmd
Device mapping:
/job:localhost/replica:0/task:0/gpu:0 -> device: 0, name: Tesla K20m, pci bus
id: 0000:02:00.0
/job:localhost/replica:0/task:0/gpu:1 -> device: 1, name: Tesla K20m, pci bus
id: 0000:03:00.0
/job:localhost/replica:0/task:0/gpu:2 -> device: 2, name: Tesla K20m, pci bus
id: 0000:83:00.0
/job:localhost/replica:0/task:0/gpu:3 -> device: 3, name: Tesla K20m, pci bus
id: 0000:84:00.0
Const_3: /job:localhost/replica:0/task:0/gpu:3
Const_2: /job:localhost/replica:0/task:0/gpu:3
MatMul_1: /job:localhost/replica:0/task:0/gpu:3
Const_1: /job:localhost/replica:0/task:0/gpu:2
Const: /job:localhost/replica:0/task:0/gpu:2
MatMul: /job:localhost/replica:0/task:0/gpu:2
AddN: /job:localhost/replica:0/task:0/cpu:0
[[  44.   56.]
 [  98.  128.]]
```

[cifar10 tutorial](http://wiki.jikexueyuan.com/project/tensorflow-zh/how_tos/tensorflow-zh/SOURCE/tutorials/deep_cnn/index.html) 这个例子很好的演示了怎样用GPU集群训练.

本节全文直接拷贝于[使用 GPUs](http://wiki.jikexueyuan.com/project/tensorflow-zh/how_tos/using_gpu.html)





## OP(operation)

OP表示某种抽象计算，它拥有0个或多个「输入/输出」，及其0个或多个「属性」。其中，输入/输出以Tensor的形式存在。



在Graph中可以通过函数get_operations()获取该图的OP信息：

```
graph = tf.Graph()
names = [op.name for op in model.graph.get_operations() if op.type=='Conv2D']
```



### 增加一个新 Op

如果现有的库没有涵盖你想要的操作, 你可以自己定制一个. 为了使定制的 Op 能够兼容原有的库 , 你必须做以下工作:

-   在一个 C++ 文件中注册新 Op. Op 的注册与实现是相互独立的. 在其注册时描述了 Op 该如何执行. 例如, 注册 Op 时定义了 Op 的名字, 并指定了它的输入和输出.
-   使用 C++ 实现 Op. 每一个实现称之为一个 "kernel", 可以存在多个 kernel, 以适配不同的架构 (CPU, GPU 等)或不同的输入/输出类型.
-   创建一个 Python 包装器（wrapper）. 这个包装器是创建 Op 的公开 API. 当注册 Op 时, 会自动生成一个默认 默认的包装器. 既可以直接使用默认包装器, 也可以添加一个新的包装器.
-   (可选) 写一个函数计算 Op 的梯度.
-   (可选) 写一个函数, 描述 Op 的输入和输出 shape. 该函数能够允许从 Op 推断 shape.
-   测试 Op, 通常使用 Pyhton。如果你定义了梯度，你可以使用Python的[GradientChecker](https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/python/kernel_tests/gradient_checker.py)来测试它。



具体流程参考 [增加一个新 Op](http://wiki.jikexueyuan.com/project/tensorflow-zh/how_tos/adding_an_op.html)  以及 [官网链接]( <https://www.tensorflow.org/extend/adding_an_op )。





## Variable

### tf.placeholder()

`tf.placeholder(dtype, shape=None, name=None)` 

- dtype：数据类型。常用的是`tf.float32` , `tf.float64` 等数值类型
- shape：数据形状。默认是None，就是一维值，数量任意。也可以是多维，比如[2, 3], [None, 3]表示列是3，行不定
- name：名称。

占位符，可以类比于形参，用于定义过程，在执行的时候`sess.run([..], feed_dict = {})` 再赋具体的值。



### tf.Variable()

`tf.Variable.init(initial_value, trainable=True, collections=None, validate_shape=True, name=None)`

| 参数名称           | 参数类型             | 含义                                       |
| -------------- | ---------------- | ---------------------------------------- |
| initial_value  | 所有可以转换为Tensor的类型 | 变量的初始值                                   |
| trainable      | bool             | 如果为True，会把它加入到GraphKeys.TRAINABLE_VARIABLES，才能对它使用Optimizer |
| collections    | list             | 指定该图变量的类型、默认为[GraphKeys.GLOBAL_VARIABLES] |
| validate_shape | bool             | 如果为False，则不进行类型和维度检查                     |
| name           | string           | 变量的名称，如果没有指定则系统会自动分配一个唯一的值               |

虽然有一堆参数，但只有第一个参数initial_value是必需的，用法如下（assign函数用于给图变量赋值）：

~~~ python
#create  a variable with a random value.  
weights=tf.Variable(tf.random_normal([5,3],stddev=0.35),name="weights")  
#Create another variable with the same value as 'weights'.  
w2=tf.Variable(weights.initialized_value(),name="w2")  
#Create another variable with twice the value of 'weights'  
w_twice=tf.Variable(weights.initialized_value()*0.2, name="w_twice") 

v = tf.Variable(3, name='v')

init=tf.global_variables_initializer()  
with tf.Session() as sess:  
    sess.run(init)  
    weights_val, w2_val, w_twice_val=sess.run([weights,w2,w_twice])  
    print(weights_val)
    print(w2_val)
    print(w_twice_val)  
    print(sess.run(v))
~~~



在TensorFlow的世界里，变量的定义和初始化是分开的，所有关于图变量的赋值和计算都要通过tf.Session的run来进行。要将所有图变量进行集体初始化时应使用tf.global_variables_initializer()[^1]  。

**但有时一个变量的初始化依赖于其他变量的初始化，但是为了确保初始化顺序不能错，可以使用initialized_value()**。

变量初始化代码

```python
tf.global_variables_initializer() 
with tf.Session() as sess:
	sess.run(init)
或者
tf.Session().run(tf.global_variables_initializer())
```

tensorflow支持有选择的初始化变量，例如：

``` python
init_new_vars_op = tf.initialize_variables([v_6, v_7, v_8])
sess.run(init_new_vars_op)
```

识别未初始化的变量

``` python
uninit_vars = []
for var in tf.all_variables():
    try:
        sess.run(var)
    except tf.errors.FailedPreconditionError:
        uninit_vars.append(var)

init_new_vars_op = tf.initialize_variables(uninit_vars)
```

以上参考 [TensorFlow 学习（三）](http://blog.csdn.net/lanchunhui/article/details/61926516)



### tf.get_variable

`tf.get_variable(name, shape, initializer)` 

-   `name` 就是变量的名称，
-   `shape` 是变量的维度，
-   `initializer`是变量初始化的方式。

### 变量初始化

- tf.constant_initializer：常量初始化函数

  `tf.constant(value, dtype=None, shape=None, name='Const')` 

  将变量初始化为给定的常量,初始化一切所提供的值。


- tf.random_normal_initializer：正态分布

  `tf.random_normal_initializer(mean,stddev)`

  功能是将变量初始化为满足正太分布的随机值，主要参数（正太分布的均值和标准差），用所给的均值和标准差初始化均匀分布

- tf.truncated_normal_initializer：截取的正态分布

  `tf.truncated_normal_initializer(mean,stddev,seed,dtype)`

  功能：将变量初始化为满足正太分布的随机值，但如果随机出来的值偏离平均值超过2个标准差，那么这个数将会被重新随机

- tf.random_uniform_initializer：均匀分布

  `tf.random_uniform_initializer(a,b,seed,dtype)` 

  从a到b均匀初始化，将变量初始化为满足平均分布的随机值，主要参数（最大值，最小值）

- tf.uniform_unit_scaling_initializer：满足均匀分布，但不影响输出数量级的随机值

  `tf.uniform_unit_scaling_initializer(factor,seed,dtypr)`   

  将变量初始化为满足平均分布但不影响输出数量级的随机值

- tf.zeros_initializer：全部是0，可简写为tf.Zeros()

  `tf.Zeros(shape, dtype=tf.float32, name=None)` 

- tf.ones_initializer：全是1，可简写为tf.Ones()

  `tf.Ones(shape, dtype=tf.float32, name=None)`

- `tf.zeros_like(tensor, dtype=None, name=None)` 

- `tf.ones_like(tensor, dtype=None, name=None)` 

- `tf.fill(dims, value, name=None)` 

  创建一个维度为dims，值为value的tensor对象．该操作会创建一个维度为dims的tensor对象，并将其值设置为value




自定义初始化，可以自己写参数初始化函数，比如：

~~~ python
def kaiming(shape, dtype, partition_info=None):
  """Kaiming initialization as described in https://arxiv.org/pdf/1502.01852.pdf
  Args
    shape: dimensions of the tf array to initialize
    dtype: data type of the array
    partition_info: (Optional) info about how the variable is partitioned.
      See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/init_ops.py#L26
      Needed to be used as an initializer.
  Returns
    Tensorflow array with initial weights
  """
  return(tf.truncated_normal(shape, dtype=dtype)*tf.sqrt(2/float(shape[0])))

 w = tf.get_variable(name="w1", initializer=kaiming, shape=[1, 10], dtype=tf.float32)
 b = tf.get_variable(name="b1", initializer=kaiming, shape=[10], dtype=tf.float32)
~~~



参考链接:

  1. [Tensorlow 中文API](http://blog.csdn.net/hk_john/article/details/78189676) 

2. [TensorFlow图变量tf.Variable的用法解析](http://blog.csdn.net/gg_18826075157/article/details/78368924)

  ​

### 变量共享

变量共享的场景主要存在于为了减少需要训练参数的个数，或是多机多卡并行化训练大数据大模型(比如数据并行化)等情况。tensorflow主要通过tf.variable_scope()， tf.name_scope(), tf.get_variable()这几个函数来实现。

我们可以通过tf.Variable()来新建变量，但是，在tensorflow程序中，我们又需要共享变量（share variables），于是，就有了tf.get_variable()(新建变量或者取已经存在的变量)。但是，因为一般变量命名比较短，那么，此时，我们就需要类似目录工作一样的东西来管理变量命名，于是，就有了tf.variable_scope()，同时，设置reuse标志，就可以来决定tf.get_variable()的工作方式（新建变量或者取得已经存在变量）。此外，在tensorflow中，还存在ops操作，而tf.variable_scope()可以同时对variable和ops的命名有影响，即加前缀；而tf.name_scope()只能对ops的命名加前缀。

- **tf.get_variable() 和 tf.Variable()**

  tf.Variable()和tf.get_variable()都是用于在一个name_scope下面获取或创建一个变量的两种方式，区别在于：

  - tf.Variable()用于创建一个新变量，在同一个name_scope下面，可以创建相同名字的变量，底层实现会自动引入别名机制，两次调用产生了其实是两个不同的变量。
  - tf.get_variable()用于获取一个变量，并且不受name_scope的约束。当这个变量已经存在时，则自动获取；如果不存在，则自动创建一个变量。
  - tf.get_variable(<name>, <shape>, <initializer>)需要初始化

  tf.get_variable()拥有一个变量检查机制，会检测已经存在的变量是否设置为共享变量，如果已经存在的变量没有设置为共享变量，TensorFlow 运行到第二个拥有相同名字的变量的时候，就会报错。

  ``` python
  def my_image_filter(input_images):
      conv1_weights = tf.Variable(tf.random_normal([5, 5, 32, 32]),
          name="conv1_weights")
      conv1_biases = tf.Variable(tf.zeros([32]), name="conv1_biases")
      conv1 = tf.nn.conv2d(input_images, conv1_weights,
          strides=[1, 1, 1, 1], padding='SAME')
      return  tf.nn.relu(conv1 + conv1_biases)

  result1 = my_image_filter(image1)
  result2 = my_image_filter(image2)

  vars = tf.all_variables()
  for v in vars:
      print(v.name)
      
  """  OUTPUT
      conv1_weights:0
      conv1_biases:0
      conv1_weights_1:0
      conv1_biases_1:0
  """
  ```

  上述代码，使用tf.Variable() 创建代码，调用两次没有出现问题，但是会生成两套变量名，因为没有共享。

  但如果把tf.Variable 改成 tf.get_variable，直接调用两次，就会出问题了。

  ```python
  result1 = my_image_filter(image1)
  result2 = my_image_filter(image2)
  # Raises ValueError(... conv1/weights already exists ...)
  ```

  为了解决这个问题，TensorFlow 又提出了 tf.variable_scope 函数：它的主要作用是，在一个作用域 scope 内共享一些变量.

- **tf.variable_scope()和tf.name_scope()**

  name_scope 只能管住操作 Ops 的名字，而管不住变量 Variables 的名字。

  name_scope 作用于操作，variable_scope 可以通过设置reuse 标志以及初始化方式来影响域下的变量。

  ~~~ python
  with tf.variable_scope("foo"):
      with tf.name_scope("bar"):
          v = tf.get_variable("v", [1])
          x = 1.0 + v
  assert v.name == "foo/v:0"
  assert x.op.name == "foo/bar/add"
  ~~~

  ​

需要注意的是：

- 创建一个新的variable_scope时不需要把reuse属性设置未False，只需要在使用的时候设置为True就可以了。
- variable_scope同样会对tf.Variable产生影响，即加上名称前缀，但多次调用会自动生成多套变量名，因为没有变量共享。而tf.get_variable方式创建的变量可以通过在variable_scope中传入reuse=True实现变量共享。

>  参考链接的第一篇[共享变量](http://wiki.jikexueyuan.com/project/tensorflow-zh/how_tos/variable_scope.html)很好，看它就够了。

参考链接：

1. [共享变量 ](http://wiki.jikexueyuan.com/project/tensorflow-zh/how_tos/variable_scope.html)
2. [tf.variable_scope(),tf.name_scope(),tf.get_variable()的认识（补充）](http://blog.csdn.net/IB_H20/article/details/72936574)
3. [tensorflow里面name_scope, variable_scope等如何理解？ - C Li的回答 - 知乎](https://www.zhihu.com/question/54513728/answer/181819324)






## Tensor

TensorFlow用`Tensor` 这种数据结构来表示所有的数据。你可以把一个张量想象成一个n维的数组或列表,非常类似于Numpy的`ndarray` 。一个张量有一个静态类型和动态类型的维数。张量可以在图中的节点之间流通。

在TensorFlow系统中，张量的维数来被描述为*阶*.但是张量的阶和矩阵的阶并不是同一个概念.张量的阶（有时是关于如*顺序*或*度数*或者是*n维*）是张量维数的一个数量描述.比如，下面的张量（使用Python中list定义的）就是2阶.

```
    t = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
```

你可以认为一个二阶张量就是我们平常所说的矩阵，一阶张量可以认为是一个向量.对于一个二阶张量你可以用语句`t[i, j]`来访问其中的任何元素.而对于三阶张量你可以用 `t[i, j, k]`  来访问其中的任何元素.

| 阶    | 数学实例        | Python 例子                                |
| ---- | ----------- | ---------------------------------------- |
| 0    | 纯量 (只有大小)   | `s = 483`                                |
| 1    | 向量(大小和方向)   | `v = [1.1, 2.2, 3.3]`                    |
| 2    | 矩阵(数据表)     | `m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]`  |
| 3    | 3阶张量 (数据立体) | `t = [[[2], [4], [6]], [[8], [10], [12]], [[14], [16], [18]]]` |
| n    | n阶 (自己想想看)  | `....`                                   |

### 形状

TensorFlow文档中使用了三种记号来方便地描述张量的维度：阶，形状以及维数.下表展示了他们之间的关系：

| 阶    | 形状               | 维数   | 实例                          |
| ---- | ---------------- | ---- | --------------------------- |
| 0    | [ ]              | 0-D  | 一个 0维张量. 一个纯量.              |
| 1    | [D0]             | 1-D  | 一个1维张量的形式[5].               |
| 2    | [D0, D1]         | 2-D  | 一个2维张量的形式[3, 4].            |
| 3    | [D0, D1, D2]     | 3-D  | 一个3维张量的形式 [1, 4, 3].        |
| n    | [D0, D1, ... Dn] | n-D  | 一个n维张量的形式 [D0, D1, ... Dn]. |

形状可以通过Python中的整数列表或元祖（int list或tuples）来表示，也或者用[`TensorShape` class](http://wiki.jikexueyuan.com/project/tensorflow-zh/api_docs/python/framework.html).



### 数据类型

除了维度，Tensors有一个数据类型属性.你可以为一个张量指定下列数据类型中的任意一个类型：

| 数据类型           | Python 类型      | 描述                         |
| -------------- | -------------- | -------------------------- |
| `DT_FLOAT`     | `tf.float32`   | 32 位浮点数.                   |
| `DT_DOUBLE`    | `tf.float64`   | 64 位浮点数.                   |
| `DT_INT64`     | `tf.int64`     | 64 位有符号整型.                 |
| `DT_INT32`     | `tf.int32`     | 32 位有符号整型.                 |
| `DT_INT16`     | `tf.int16`     | 16 位有符号整型.                 |
| `DT_INT8`      | `tf.int8`      | 8 位有符号整型.                  |
| `DT_UINT8`     | `tf.uint8`     | 8 位无符号整型.                  |
| `DT_STRING`    | `tf.string`    | 可变长度的字节数组.每一个张量元素都是一个字节数组. |
| `DT_BOOL`      | `tf.bool`      | 布尔型.                       |
| `DT_COMPLEX64` | `tf.complex64` | 由两个32位浮点数组成的复数:实数和虚数.      |
| `DT_QINT32`    | `tf.qint32`    | 用于量化Ops的32位有符号整型.          |
| `DT_QINT8`     | `tf.qint8`     | 用于量化Ops的8位有符号整型.           |
| `DT_QUINT8`    | `tf.quint8`    | 用于量化Ops的8位无符号整型.           |

原文来自: [Tensor Ranks, Shapes, and Types](http://www.tensorflow.org/resources/dims_types.md)

翻译来自: [张量的阶、形状、数据类型](http://wiki.jikexueyuan.com/project/tensorflow-zh/resources/dims_types.html) 






## TFRecords

TFRecords是TF官方推荐使用的数据存储形式

相比于直接用python文件读取图片，如果多次读取小文件还是推荐使用TFRecords，速度更快。但是究竟快多少呢？[这里我对这两种方法做了对比](https://zhuanlan.zhihu.com/p/27481108)。



这里有一篇踩坑之路,里面介绍的经验很好。

[使用 Tensorflow 读取大数据时的踩坑之路](https://zhuanlan.zhihu.com/p/28450111) 



TensorFlow1.5之后可以使用新的[DatasetAPI去读取数据](https://zhuanlan.zhihu.com/p/34020748)。

参考链接:

[tensorflow TFRecords文件的生成和读取方法](https://zhuanlan.zhihu.com/p/31992460)

[使用自己的数据集进行一次完整的TensorFlow训练](https://zhuanlan.zhihu.com/p/32490882)

[tensorflow数据的读取](https://zhuanlan.zhihu.com/p/34518755)





## TensorBoard

可以参考一下两篇链接

1.  [TensorBoard:可视化学习](http://wiki.jikexueyuan.com/project/tensorflow-zh/how_tos/summaries_and_tensorboard.html)
2.  [TensorBoard: 图表可视化](http://wiki.jikexueyuan.com/project/tensorflow-zh/how_tos/graph_viz.html)





## TensorFlowLite

注意: 这个功能需要tensorflow1.5以上版本才支持。

 待续。。



## 超参数设置

### 学习率

-   指数衰减学习率

```python
tf.train.exponential_decay(
 learning_rate,   # 初始学习率
 global_step, 	  # 当前的学习步数，等同于我们将 batch 放入学习器的次数
 decay_steps,     # 每轮学习的步数，decay_steps=sample_size/batch 即样本总数除以每个batch的大小
 decay_rate, 	  # 每轮学习的衰减率，0<decay_rate<1
 staircase=False, # 如果True，那么global_step/decay_steps是一个整数除法，衰减的学习率遵循阶梯函数。
 name=None
)
```

当前学习率`lr_rate` 等于:

$ lr\_rate = learning\_rate*decay\_rate^\frac{global\_step}{decay\_steps}$   

**注意** :  这些参数类型都为tensorflow的tensor格式

### 优化器

`tf.train.AdamOptimizer` 

``` python
tf.train.AdamOptimizer.__init__(
    learning_rate=0.001, 
    beta1=0.9, 
    beta2=0.999, 
    epsilon=1e-08, 
    use_locking=False, 
    name='Adam'
)
```



参考[这篇文档](http://www.tensorfly.cn/tfdoc/api_docs/python/train.html)





## API 

### 图像操作

- `tf.slice`  图像裁剪
- `tf.transpose` 图像翻转

### tf.convert_to_tensor() 

可将 numpy 下的多维数组转化为 tensor。

### tf.assign()

赋值函数，比如tf.assign(A, new_number): 这个函数的功能主要是把A的值变为new_number，[参见](http://blog.csdn.net/uestc_c2_403/article/details/72235310) .

同时该函数可以改变变量的type和shape。



### tf.trainable_variables()

获得训练中可学习的参数变量, 用关键字trainable=False来控制。

```  python
v = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32), name='v')  
v1 = tf.Variable(tf.constant(5, shape=[1], dtype=tf.float32), name='v1')  
  
global_step = tf.Variable(tf.constant(5, shape=[1], dtype=tf.float32), name='global_step', trainable=False)  
```

### tf.global_variables()[^2]  

返回所有变量的列表

### tf.reduce_mean()

`tf.reduce_mean(input_tensor, reduction_indices=None, keep_dims=False, name=None)` 

功能: 求Tensor平均值

参数1--input_tensor:待求值的tensor。

参数2--reduction_indices:在哪一维上求解。

参数（3）（4）可忽略

~~~ python
# 'x' is [[1., 2.]
#         [3., 4.]]

tf.reduce_mean(x) ==> 2.5 #如果不指定第二个参数，那么就在所有的元素中取平均值
tf.reduce_mean(x, 0) ==> [2.,  3.] #指定第二个参数为0，则第一维的元素取平均值，即每一列求平均值
tf.reduce_mean(x, 1) ==> [1.5,  3.5] #
~~~

### tf.reduce_max()/tf.reduce_min()/tf.reduce_sum()

同 tf.reduce_mean()

### tf.argmax ()/tf.argmin()

返回最大值/最小值所在的坐标

### tf.reduce_all() /tf.reduce_any()

逻辑运算

计算tensor中各个元素的逻辑和（and运算）/（or运算）

~~~ python
# 'x' is [[True, True] 
# [False, False]] 
tf.reduce_all(x) ==> False 
tf.reduce_all(x, 0) ==> [False, False] 
tf.reduce_all(x, 1) ==> [True, False]
~~~

### tf.cast()

类型转换， 比如 tf.int32 --> tf.float32

### tf.equal()

比较两个tensor的值，如果在一个下表下一样，那么返回的tensor在这个下表上就为true.这个函数一般可以与cast在一起去（cast到float32上）计算一些准确率

### tf.clip_by_norm()

`clip_by_norm(t, clip_norm, axes=None, name=None)`

clip_by_norm是指对梯度进行裁剪，通过控制梯度的最大范式，防止梯度爆炸的问题，是一种比较常用的梯度规约的方式。 

> Specifically, in the default case where all dimensions are used for calculation, if the L2-norm of `t` is already less than or equal to `clip_norm`, then `t` is not modified. If the L2-norm is greater than `clip_norm`, then this operation returns a tensor of the same type and shape as `t` with its values set to: `t * clip_norm / l2norm(t)` In this case, the L2-norm of the output tensor is `clip_norm`.





---

[^1]: ~~tf.initialize_all_variables()~~ 该函数将不再使用。 
[^2]: ~~tf.all_variables()~~ 不再使用



[^_^]: H~2~O  X^2^  注释 如果脚注不被引用是不会显示出来的
