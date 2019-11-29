##
# Script to change (.tif) image slices into 3D volumes in Nifti format. Are nicer to work with, easy numpy compatibility
# preservation of slice order etc.
# see main method to change paths and whether you work with segmentations or not, works with different datasets, not just one
##

import numpy as np
import nibabel as nib
import re
from create_h5_dataset import get_file_list
import cv2
import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def read_save_files(dimensions, file_list, file_name, segment):
    """
    Function takes dimensions and file_list as input (obtained from other helper function) and saves all images
    in a certain folder into a 3D volume image
    :param dimensions: image dimensions (usually: [X, Y, 1] as we're dealing with grayscale images
    :param file_list: list of files in a folder
    :param segment: Boolean indicating if input image is a GT segmentation (modified for heart input, might have to tune for other applications)
    :return:
    """

    im_list = []
    regex = re.compile(r'\d+')
    file_list = file_list
    vol_arr = np.zeros((len(file_list), dimensions[0], dimensions[1], dimensions[2]), dtype=np.uint8)
    print('INFO:   Shape of 3D Image: '+ str(vol_arr.shape))

    if segment:
        print('INFO:   Segmented images as input, converting values to get better contrast (ONLY valid for heart data)')
    else:
        print('INFO:   Raw images as input, no value conversion done')

    for el in file_list:
        reg_list = regex.findall(el)
        im_list.append(reg_list[-1])
        img = np.array(cv2.imread(el, cv2.IMREAD_GRAYSCALE))
        if segment:
            img[img == 1] = 0
            img[img == 2] = 120
            img[img == 3] = 255

        img = img.reshape((dimensions[0], dimensions[1], dimensions[2]))
        vol_arr[int(reg_list[-1]), :, :, :] = img


    # im_list = sorted(im_list)
    nib_img = nib.Nifti1Image(vol_arr, np.eye(4))
    nib.save(nib_img, save_path + file_name + '.nii.gz')
    print('INFO:   Saved 3D image into: ' + save_path + ' as: ' + file_name)


if __name__ == '__main__':


    ## Raw images
    segment = False
    data_path = './Data/Heart/Raw/'
    save_path = './Data/Heart/3D/Raw/'

    ## Segmented images
    # segment = True
    # data_path = './Data/Heart/Segmented/'
    # save_path = './Data/Heart/3D/Segmented/'

    files = os.listdir(data_path)
    files.remove('old')

    filepaths = []
    filenames = []

    for el in files:
        filepaths.append(data_path + el + '/')
        filenames.append(el)

    assert len(filepaths) == len(filenames)

    for idx in range(len(filepaths)):
        im_path = filepaths[idx]
        file_name = filenames[idx]
        file_list, dimensions, flag = get_file_list(im_path)
        im_list = read_save_files(dimensions, file_list, file_name, segment)
