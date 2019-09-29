---
layout:     post
title:      "TensorFlow Saver Tips"
subtitle:   "TensorFlow 加载权值的一些总结"
date:       2019-04-12
author:     "西轩"
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - TensorFlow
    - Saver
    - Tips
---



## TensorFlow Saver



### 只加载权值

``` python
with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, "./checkpoint/finetune")
```



### 加载图结构和权值

``` python
with tf.Session() as sess:
	saver = tf.train.import_meta_graph('./checkpoint/finetune.meta')
    saver.restore(sess, "./checkpoint/finetune")
```



### 加载部分权值

 switch the optimizer from `rmsprop` to `adam`, 加载权值失败

[stackoverflow 相似问题](https://stackoverflow.com/questions/47194449/changing-optimizer-results-in-not-found-in-checkpoint-errors) 

``` python
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    all_variables = tf.global_variables()
    variables_to_restore = []
    for var in all_variables:
        # if 'Momentum' in var.name:
        if 'Adam' in var.name or '_power' in var.name:
            print("Ignore ", var.name)
            continue
        variables_to_restore.append(var)
    
    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess, "./checkpoint/pre_model/finetune")
```



### 冻结权值，保存成PB

```python
from tensorflow.python.framework import graph_util

# 输出结点
pred_classes = tf.argmax(pred, axis=1, name="output_cls")

constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['output_cls'])
```



### 跑PB文件

```python
from tensorflow.python.platform import gfile

sess = tf.Session()
with gfile.FastGFile('./checkpoint/model.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='') # 导入计算图

# 需要有一个初始化的过程    
sess.run(tf.global_variables_initializer())


# 输入
input_ = sess.graph.get_tensor_by_name('input:0')
prob_ = sess.graph.get_tensor_by_name('kepp_prob:0')

rslt = sess.graph.get_tensor_by_name('output_cls:0')

ret = sess.run(rslt,  feed_dict={input_: mnist.test.images[:2], prob_: 1.0})
```

