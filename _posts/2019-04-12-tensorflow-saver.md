---
layout:     post
title:      "TensorFlow Saver Tips"
subtitle:   "TensorFlow 加载权值的一些总结"
date:       2020-05-26
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



### 获得Weight的值

```python
import tensorflow as tf

# for checkpoint
for tv in tf.trainable_variables():
    print (tv.name)
b = tf.get_default_graph().get_tensor_by_name("generate/resnet_stack/bias:0")
w = tf.get_default_graph().get_tensor_by_name("generate/resnet_stack/weight:0")


# for pb
# https://stackoverflow.com/questions/35336648/list-of-tensor-names-in-graph-in-tensorflow
sess = tf.Session()
op = sess.graph.get_operations()
[m.values() for m in op][1]

out:
(<tf.Tensor 'conv1/weights:0' shape=(4, 4, 3, 32) dtype=float32_ref>,)

with tf.Session() as sess:
        with gfile.FastGFile(pb_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')

            op = sess.graph.get_operations()
            for m in op:
                if 'weights' in m.name:
                    print(m.name)
                    val = m.values()[0]
                    val_np = sess.run(val)
                pass
            # [m.values() for m in op][1]


# for tflite
# https://stackoverflow.com/questions/52111699/how-can-i-view-weights-in-a-tflite-file
from tflite import Model
buf = open('/path/to/mode.tflite', 'rb').read()
model = Model.Model.GetRootAsModel(buf, 0)
subgraph = model.Subgraphs(0)
# Check tensor.Name() to find the tensor_idx you want
tensor = subgraph.Tensors(tensor_idx) 
buffer_idx = tensor.Buffer()
buffer = model.Buffers(buffer_idx)
# After that you'll be able to read the data by calling buffer.Data()
```



### 修改Weight的值

``` python
all_vars = tf.global_variables()
for var in all_vars:
	sess.run(var.assign(val))  # val is numpy.ndarray
```



### 模型转换 （checkpoints，pb, tflite）

```python
import sys
import numpy as np
import tensorflow as tf

if not sys.version_info[1] == 5:
    raise Exception('It must be Python 3.5')
    
# 关闭tf c++ 输出
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#  remove the disgusting "deprecated" warning messages
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def save_pb(model_path, pb_path, quantize=False):
    """
    @pargm: model_path='./checkpoint/quantize/model'
    @pargm: pb_path='./checkpoint/quantize/model.pb'
    """
    from tensorflow.python.framework import graph_util
    _inptus = tf.placeholder(tf.float32, [1, 112, 112, 3], name='inputs')
    # _is_training = True if is_quantize else tf.placeholder(tf.bool, name="is_training")
    rslt = build_ResNet(_inptus, False)
    if quantize:
        tf.contrib.quantize.create_eval_graph()

    device_count = {"GPU": 0}
    with tf.Session(config=tf.ConfigProto(device_count=device_count)) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        print("save model")
        if quantize:
            constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['resnet/fc/output/act_quant/FakeQuantWithMinMaxVars'])
        else:
            constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['resnet/fc/output/BiasAdd'])
        with tf.gfile.FastGFile(model_path + ".pb", mode='wb') as f:
            f.write(constant_graph.SerializeToString())


def val_pb(pb_path):
    from tensorflow.python.platform import gfile
    tf.reset_default_graph()
    
    # data
    input_data = np.random.rand(...)
    print(input_data.shape)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        with gfile.FastGFile(pb_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
            _inputs = sess.graph.get_tensor_by_name('inputs:0')
            rslt = sess.graph.get_tensor_by_name('resnet/fc/output/BiasAdd:0')

        rslt_ = sess.run(rslt, feed_dict={_inputs: input_data})
        # print(rslt_)
    pass


def save_lite(pb_path, is_quantize=False):
    """
    r1.14
    https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter
    """
    print("INFO: save tflite")
    converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph(
    # converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        pb_path, ['inputs'], ['resnet/fc/output/BiasAdd'])
    if is_quantize:
        converter.inference_type = tf.contrib.lite.constants.QUANTIZED_UINT8
        converter.quantized_input_stats = {'Placeholder': (0., 255.)}
    tflite_model = converter.convert()
    lite_path = pb_path.replace(".pb", ".tflite")
    open(lite_path, "wb").write(tflite_model)
    pass


def val_lite(lite_path):
    print("INFO: verify tflite")
    interpreter = tf.contrib.lite.Interpreter(model_path=lite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("input_details:")
    print(input_details[0])
    # print(input_details[1])
    print("output_details:")
    print(output_details[0])

    # idx = input_details[0]['index']
    # print("input_details index: %s" % idx)
    # idx = input_details[1]['index']
    # print("input_details index: %s" % idx)
    # # Test model on random input data.
    # input_shape = input_details[0]['shape']
    # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    # interpreter.set_tensor(input_details[0]['index'], input_data)
    # input_shape = input_details[1]['shape']
    # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    # interpreter.set_tensor(input_details[1]['index'], input_data)

    # print(f"input_c :\n {input_c.shape} {input_c.dtype}")
    # print(f"input_d :\n {input_d.shape} {input_d.dtype}")

    inputs = np.ones((1, 112, 112, 3)).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], inputs)
    # interpreter.set_tensor(input_details[1]['index'], input_d)
    interpreter.invoke()
    rslt_ = interpreter.get_tensor(output_details[0]['index'])
    print("rslt_ shape: {}, \n {}".format(rslt_.shape, rslt_))
    pass

```



