import numpy as np
# import tensorflow as tf
# from tensorflow.python.client import device_lib
import os
# os.environ["CUDA_VISIBLE_DEVICES"]= "0"
#
# print(tf.__version__)
# device_lib.list_local_devices()
# tf.test.is_gpu_available()
# tf.test.gpu_device_name()
#
#
# a = tf.constant(2.0)
# b = tf.constant(3.0)
#
# c = tf.add(a,b)
#
# with tf.Session() as sess:
#     print(sess.run(c))

gauss = np.array((-1.0, 0.0, 0.5, 1.5)).reshape(2,2)
print(gauss)
# for element in np.nditer(gauss):
# with np.nditer(gauss, op_flags=['readwrite']) as it:
#     for element in it:
#         if element[...] >= 1.0:
#             element[...] = 1.0
#         elif element[...] < 0.0:
#             element[...] = 0.0

gauss[gauss > 1.0] = 1
gauss[gauss < 0.0] = 0
print(gauss)