import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

print(tf.__version__)
device_lib.list_local_devices()
tf.test.is_gpu_available()
tf.test.gpu_device_name()


a = tf.constant(2.0)
b = tf.constant(3.0)

c = tf.add(a,b)

with tf.Session() as sess:
    print(sess.run(c))

