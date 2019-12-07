import numpy as np
import test2
# import tensorflow as tf
# from tensorflow.python.client import device_lib
import os
import re
import sys
import nibabel as nib
import logging
import scipy
import imageio as io
import h5py
import cv2
from scipy.misc import imshow
from scipy.misc import toimage
from sklearn.metrics import f1_score, roc_auc_score
from scipy.stats import pearsonr
#######################################################################################################################
# os.environ["CUDA_VISIBLE_DEVICES"]= "0"
#
# print(tf.__version__)
# device_lib.list_local_devices()
# tf.test.is_gpu_available()
# tf.test.gpu_device_name()
#
#
# logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
#######################################################################################################################

# data_path = './Data/Heart/3D/Segmented/'
# filename = '06_WK1_03_Segm_3D.nii.gz'
#
# raw_data_path = './Data/Heart/3D/Raw/'
#
# raw_list = []
# syn_list = []
#
# files = os.listdir(data_path)
# raw_files = os.listdir(raw_data_path)
#
# for el in os.listdir(data_path):
#     syn_list.append(data_path + el)
#
# for el in os.listdir(raw_data_path):
#     raw_list.append(raw_data_path + el)
#
# print('seg files: ', syn_list)
# print('raw files: ',raw_list)
# print('')
# print('')
# print('sorted seg files: ', sorted(syn_list))
# print('sorted raw files: ', sorted(raw_list))

#######################################################################################################################

# path = './Data/Heart/3D/Raw/06_WK1_03_Cropabs.nii.gz'
# save_path = './Data/Heart/Test/'
# raw_img = nib.load(path)
# raw_data = raw_img.get_fdata()
#
# for ind in range(raw_data.shape[0]):
#     io.imsave(save_path + str(ind) + '.png', raw_data[ind,:,:,0])

#######################################################################################################################

# split_path = './Heart/heart_ids.txt'
# data_path = './Data/Heart/3D/Raw/'
# regex = re.compile(r'\d+')
#
# ind_list = []
# for el in os.listdir(data_path):
#     reg_list = regex.findall(el)
#
#     ind_list.append(int(reg_list[-1]))
#
# labels = np.array(sorted(ind_list, reverse=True))
# print(labels)
# ind = np.arange(1,12)
# labels = np.array([3,3,3,3,2,2,2,2,1,1,1])
# print(ind)
# print(labels)

# path = './data_processing/aug_heart_data_noisy.h5'
# data = h5py.File(path, 'r')
# #
# part_a = data['A/data_1'][0,20:40,20:40,0]
# part_b = data['B/data_1'][0,20:40,20:40,0]
#
# print(part_a)
# print(part_b)
# aug_factor = int(np.array(data['A/aug_factor']))
# imshape = np.shape(data['A/data_1'][:,0,0,0])[0]
# print(imshape)
#
# imsize = np.shape(data['A/data_1'][0,:,0,0])[0]
# print(imsize)
# a_chan = int(np.array(data['A/num_channel']))
# b_chan = int(np.array(data['B/num_channel']))
# print(a_chan)
# print(b_chan)
# a_size = int(np.array(data['A/num_samples']))
# b_size = int(np.array(data['B/num_samples']))
# print(a_size)
# print(b_size)
#
# aug_factor = int(np.array(data['A/aug_factor']))
#
# imsize = np.shape(data['A/data_1'][0, :, 0, 0])[0]
# aug_nr_images = np.shape(data['A/data_1'][:,0,0,0])[0]
# nr_images = aug_nr_images / aug_factor
# imshape = np.shape(data['A/data_1'][0,:,:,:])
#
# print(aug_factor)
# print(aug_nr_images)
# print(nr_images)
# print(imshape)

# list1 = []
# list2 = []
#
#
# a = np.array([120,255,120,0,0,255,120,0,0,120,120,255,255])
# noise = np.random.normal(0, 10, size=a.shape)
# b = a + noise
# b = np.clip(b, 0, 255)
# c = np.rint(b)
# print(a)
# print(b)
# print(c)


a = np.asarray([1.4,2.6,4.7])
# b = [2,3]
c = np.asarray([1.3,2.8,5.0])
# print(b)
# b += a
# print(b)
#
print(a)
print(c)
corr, _ = pearsonr(a,c)
print(corr)
# print(corr)
# a = np.array([(0,1,2,1,3,4,5,2)])
# # b = np.array([(0,1,0,1),(3,4,5,2)])
# #
# print(a.shape[1])
# a = a.reshape(a.shape[0]*a.shape[1])
# b = b.reshape(b.shape[0]*b.shape[1])
#
# list1 += list(a)
# list2 += list(b)
#
# print(list1)
# print(list2)
#
# score = f1_score(np.asarray(list1),np.asarray(list2),average=None)
# print(score)
# nr_img = imshape / aug_factor
# print(nr_img)
#
# img_a = np.array(data['A/data_7'][13245,:,:,0])
# img_b = np.array(data['B/data_7'][13245,:,:,0])
#
# toimage(img_a).show()
# toimage(img_b).show()
#

# batch_indices = np.random.choice(np.arange(1,4), size=3, replace=False)
# print(batch_indices)
# print(int(batch_indices.shape[0]))
# for idx in batch_indices:
#     print('index: ' + str(idx))
#     print(data['A' + '/data_' + str(idx)][0,0,0,:])
#
# np.random.seed(1)
# a = np.array([1,3,5,7])
# b = np.array([7,5,4,1])
# c = np.arange(10)
# d = [a, b, c]
#
# # sample()
# # sample()
# #
# # for i in range(2):
# #     sample()
# #
# # shuffle(a)
# # shuffle(b)
# class tests:
#     def __init__(self,name):
#         name = name
#
#     def samples(self):
#         for i in range(5):
#             test2.sample(c)
#
# test = tests(name='hi')
# test.samples()
# for i in range(3):
#     test2.sample(d[i])
# np.random.shuffle(a)
# np.random.shuffle(b)
# np.random.shuffle(c)
# print(a)
# print(b)
# print(c)