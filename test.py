import numpy as np
# import tensorflow as tf
# from tensorflow.python.client import device_lib
import os
import re
import sys
import nibabel as nib
import logging
import scipy
import imageio as io
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

split_path = './Heart/heart_ids.txt'
data_path = './Data/Heart/3D/Raw/'
regex = re.compile(r'\d+')

ind_list = []
for el in os.listdir(data_path):
    reg_list = regex.findall(el)
    
    ind_list.append(int(reg_list[-1]))

labels = np.array(sorted(ind_list, reverse=True))
print(labels)
# ind = np.arange(1,12)
# labels = np.array([3,3,3,3,2,2,2,2,1,1,1])
# print(ind)
# print(labels)
