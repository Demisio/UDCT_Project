"""
For this code to run, first create a normal training set with the script 'heart_augment_loader', call the output something like 'gan_aug_data'
First, this file should contain normally augmented data. Afterwards this data will be replaced by GAN data in this script
Idea: load the above mentioned file, as well as the data augmented file. Then add the GAN data to this new file according to the sample indices defined
in the GAN file --> replace only what is necessary and keep what you want to keep (e.g. validation and test)
Do not perform this for 5 fold CV but just for one simple split to see what happens
"""
import numpy as np
import nibabel as nib
import logging
import h5py
import os

np.random.seed(42)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

fold = 3
filename = './gan_data.h5'
filename_gan_res = './../Results/Heart_very_lim_data/Images/pred_train.h5'

f_gan = h5py.File(filename_gan_res, 'r')     # open the file
f1 = h5py.File(filename, 'r+')

train_sample_list = []
for name in f_gan['A/fold_3/sample_idx']:
    train_sample_list.append(name)
print(train_sample_list)


for el in train_sample_list:
    print('INFO:   Currently at Sample volume: ' + str(el))
    data_a_gan = f_gan['A/fold_3/data_' + str(el)]
    data_b_gan = f_gan['B_fake/fold_3/data_' + str(el)]

    data_a = f1['A/data_' + str(el)]       # load the data
    data_b = f1['B/data_' + str(el)]
    data_a[...] = data_a_gan               # assign new values to data
    data_b[...] = data_b_gan  # assign new values to data

    assert np.array_equal(data_a[...], data_a_gan[...])
    assert np.array_equal(data_b[...], data_b_gan[...])

f1.close()                          # close the file
f_gan.close()
# f1 = h5py.File(filename, 'r')
# np.allclose(f1['meas/frame1/data'].value, X1)