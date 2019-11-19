
import numpy as np
import os
import cv2
from create_h5_dataset import get_file_list
import sys
import io
import matplotlib
import matplotlib.pyplot as plt

def noisy(noise_typ,image):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0.10
        sigma = 0.05
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        gauss[gauss > 1.0] = 1
        gauss[gauss < 0.0] = 0
        noisy = image + gauss
        return noisy

    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out

    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy

    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)
        noisy = image + image * gauss
        return noisy

# def sys_input():
#     if len(sys.argv[1:]) != 3:
#         print('This script recuires three arguments in order to work:')
#         print(' ')
#         print('Script aborted')
#         return -1
#
#     data_path = sys.argv[1]
#     save_path = sys.argv[2]
#     add_noise = sys.argv[3]
#     crop = sys.argv[4]
#
#     if add_noise == 'noise':
#         add_noise = True
#     elif add_noise == 'no_noise':
#         add_noise = False
#
#     if crop == 'crop':
#         crop = True
#     elif crop == 'no_crop':
#         crop = False
#
#     return data_path, save_path, add_noise, crop

def modification(save_path, add_noise, crop, segment,
                 files, dimensions, flag):

    num_samples = len(files)
    num_channel = dimensions[2]
    dtype = np.uint8

    data_A = np.zeros([num_samples, \
                       dimensions[0], \
                       dimensions[1], \
                       num_channel], dtype=dtype)

    for idx, fname in enumerate(files):
        if flag:  # This means, the images are gray scale
            # print('Images are grayscale')
            data_A[idx, :, :, 0] = np.array(cv2.imread(fname, cv2.IMREAD_GRAYSCALE))
        else:
            # print('Images are not grayscale')
            data_A[idx, :, :, :] = np.flip(np.array(cv2.imread(fname, cv2.IMREAD_COLOR)), 2)

    ## add more contrast to segmentation
    if segment:
        data_A[data_A == 1] = 0
        data_A[data_A == 2] = 120
        data_A[data_A == 3] = 255

    ## cropped data
    start = 100
    stop = start + 256
    data_A_crop = data_A[:, start:stop, start:stop, :]

    ## test
    print('Original Shape of an Image:')
    print(data_A[0].shape)
    print('')
    print('Cropped Shape of an Image:')
    print(data_A_crop[0].shape)
    print('')

    plt.imshow(data_A_crop[0])

    if add_noise and not crop:
        print('Added Noise: Yes, Cropped: No, Converted to .png: Yes')
        for index in range (data_A.shape[0]):
            # noisy
            noisy_img = noisy('gauss', data_A[index])
            cv2.imwrite(save_path + str(index) + '_noisy_' + '.png', noisy_img)

    elif not add_noise and not crop:
        print('Added Noise: No, Cropped: No, Converted to .png: Yes')
        for index in range(data_A.shape[0]):
            # #normal
            normal_img = data_A[index]
            cv2.imwrite(save_path + str(index) + '_png_' + '.png', normal_img)

    elif add_noise and crop:
        print('Added Noise: Yes, Cropped: Yes, Converted to .png: Yes')
        for index in range(data_A_crop.shape[0]):
            noisy_img = noisy('gauss', data_A_crop[index])

    elif not add_noise and crop:
        print('Added Noise: No, Cropped: Yes, Converted to .png: Yes')
        for index in range(data_A_crop.shape[0]):
            crop_img = data_A_crop[index]


if __name__ == '__main__':

    data_path = sys.argv[1]
    save_path = sys.argv[2]
    add_noise = sys.argv[3]
    crop = sys.argv[4]
    segment = sys.argv[5]

    if add_noise == 'noise':
        add_noise = True
    elif add_noise == 'no_noise':
        add_noise = False

    if crop == 'crop':
        crop = True
    elif crop == 'no_crop':
        crop = False

    if segment == 'segment':
        segment = True
    elif segment == 'no_segment':
        segment = False

    # data_path, save_path, add_noise, crop = sys_input()
    files, dimensions, flag = get_file_list(data_path)
    modification(save_path, add_noise, crop, segment,
                 files, dimensions, flag)



