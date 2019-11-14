import cv2
import h5py
import logging

##TODO: currently only tested for 2D images, not 3D images --> might have different h5py structure

'''
change these paths and names according to the experiment
'''

filename = "Models/example_model_gen_B.h5"
path = 'Images/example/'
name = 'example'
filetype = '.png'

def h5_to_img(path, name, filetype):
    with h5py.File(filename, 'r') as f:
        #display the groups --> for further subselection, here we have the group 'B'
        items = list(f.items())
        logging.info('Items: {}'.format(items))

        #display further groups or here the dataset 'data'
        G1 = f.get('B')
        G1_items = list(G1.items())
        logging.info('Items'.format(G1_items))

        #access the dataset 'data' and get some information, e.g. shape
        #dset structure: (img_index, height, width, channels)
        #access with same information --> slicing
        dset = G1.get('data')
        logging.info('Shape of dataset:'.format(dset.shape))

        data_list = []
        file_name_list = []
        for x in range(0, dset.shape[0]):
            data_list.append(dset[x])
            file_name_list.append(path + name + '_image_' + str(x) + filetype)
            # print(file_name_list[x])
            cv2.imwrite(file_name_list[x], data_list[x])

        logging.info('Length of list:', len(data_list))
        # print(file_name_list)

        logging.info('Images in hdf5 file were converted to:' + str(filetype))

if __name__ == "__main__":
    h5_to_img(path, name, filetype)