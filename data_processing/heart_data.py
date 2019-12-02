
import numpy as np
import os
from sklearn import model_selection
import h5py
import sys
from data_processing.batch_provider import BatchProvider_

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from create_h5_dataset import get_file_list

class heart_data():

    def __init__(self, path, fold):

        ## Change fold according to needs
        fold = fold
        split_path = './train_test_split/splits'
        data = h5py.File(path, 'r')

        self.data = data

        self.a_chan = int(np.array(data['A/num_channel']))      # Number channels in A
        self.b_chan = int(np.array(data['B/num_channel']))      # Number channels in B
        self.imsize = int(np.shape(data['A/data'][0,:,0,0])[0]) # Image size (squared)
        self.a_size = int(np.array(data['A/num_samples']))      # Number of samples in A
        self.b_size = int(np.array(data['B/num_samples']))      # Number of samples in B
        self.aug_factor = int(np.array(data['A/aug_factor']))   # how many times were augmentations performed? used for indexing

        self.imshape = np.shape(data['A/data_1'][0,:,:,:])      # Shape of a single image in the array
        self.aug_nr_images = np.shape(data['A/data_1'][:,0,0,0])[0]
        self.nr_images = self.aug_nr_images / self.aug_factor

        # the following are HDF5 datasets, not numpy arrays
        image_grp = data['images']
        labels = data['labels']
        feat_grp = data['features']


        train_filename = os.path.join(split_path, 'train_fold_{}.txt'.format(fold))
        total_ids_list = [int(line.split('\n')[0]) for line in open(train_filename)]
        total_ids = np.sort(np.array(total_ids_list))

        #create the split for TRAINING & VALIDATION with the indices, set 11% to test size, stratify & shuffle the split
        # ... notation to get everything in these dimensions, e.g. [1,:,:] for 3D array could be [1, ...]
        train_ids, val_ids = model_selection.train_test_split(total_ids,
                                                              test_size=0.2,
                                                              random_state=42,
                                                              shuffle=True)

        #also get indices for the TEST data.
        test_filename = os.path.join(split_path, 'test_fold_{}.txt'.format(fold))
        test_ids_list = [int(line.split('\n')[0]) for line in open(test_filename)]
        test_ids = np.sort(np.array(test_ids_list))

        # Create the batch providers
        self.train = BatchProvider_(self.data, train_ids, self.aug_factor, self.nr_images, self.imshape)
        self.validation = BatchProvider_(self.data, val_ids, self.aug_factor, self.nr_images, self.imshape)
        self.test = BatchProvider_(self.data, test_ids, self.aug_factor, self.nr_images, self.imshape)

