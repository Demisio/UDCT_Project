
import numpy as np
import os
from sklearn import model_selection
import h5py
from data_processing.batch_provider import BatchProvider_

class heart_data():

    def __init__(self, path):

        ## Change fold according to needs
        fold = 1

        data = h5py.File(path, 'r')

        self.data = data

        self.a_chan = int(np.array(data['A/num_channel']))      # Number channels in A
        self.b_chan = int(np.array(data['B/num_channel']))      # Number channels in B
        self.imsize = int(np.shape(data['A/data'][0,:,0,0])[0]) # Image size (squared)
        self.a_size = int(np.array(data['A/num_samples']))      # Number of samples in A
        self.b_size = int(np.array(data['B/num_samples']))      # Number of samples in B
        self.aug_factor = int(np.array(data['A/aug_factor']))   # how many times were augmentations performed? used for indexing

        # the following are HDF5 datasets, not numpy arrays
        image_grp = data['images']
        labels = data['labels']
        feat_grp = data['features']


        train_filename = os.path.join(exp_config.split_path, exp_config.label_name, 'train_fold_{}.txt'.format(fold))
        entire_ids = [int(line.split('\n')[0]) for line in open(train_filename)]
        unsort_entire_indices, unsort_train_img_indices = self.get_indices_from_ids(entire_ids, array_img_ids)
        entire_indices = np.sort(unsort_entire_indices)
        train_img_indices = np.sort(unsort_train_img_indices)

        #create the split for TRAINING & VALIDATION with the indices, set 11% to test size, stratify & shuffle the split
        # ... notation to get everything in these dimensions, e.g. [1,:,:] for 3D array could be [1, ...]
        train_indices, val_indices = model_selection.train_test_split(entire_indices,
                                                                      test_size=0.11,
                                                                      random_state=5,
                                                                      shuffle=True,
                                                                      stratify=labels[exp_config.label_name][entire_indices, ...])

        #also get indices for the TEST data.
        test_filename = os.path.join(exp_config.split_path, exp_config.label_name, 'test_fold_{}.txt'.format(exp_config.cv_fold))
        test_ids = [int(line.split('\n')[0]) for line in open(test_filename)]
        test_indices, test_img_indices = self.get_indices_from_ids(test_ids, array_img_ids)

        # Create the batch providers
        self.train = BatchProvider_()
        self.validation = BatchProvider_()
        self.test = BatchProvider_()


if __name__ == '__main__':
    from classifier.experiments import tavi_class as config
    data = tavi_voi_data(config)
    data.train
    data.validation.iterate_batches(32)
    data.test.iterate_batches(32)
    i = 10