import numpy as np


class BatchProvider_Heart():
    """
    Batch Provider class to provide batches of data for the network
    """
    def __init__(self, data, sample_indices, aug_factor, nr_img, imshape):
        """
        :param data: h5 data array
        :param sample_indices: Sample indices for the split
        :param aug_factor: how many times were images augmented?
        :param nr_img: the number of non-augmented images, calculated by: total_images / aug_factor
        :param imshape: shape of individual slices
        """
        self.data = data
        self.sample_indices = sample_indices
        self.aug_factor = aug_factor
        self.nr_img = nr_img
        self.imshape = imshape


    def next_batch(self, batch_size):
        """
        Get a single random batch, takes a pseudo-random slice from a random image.
        Pseudo-randomness: Slices will be from different sets if available, also since augmentation was performed,
        we have n different versions of an image, do not take the "same" image multiple times for an epoch
        """

        batch_a = np.zeros(shape=(batch_size, self.imshape[0], self.imshape[1], self.imshape[2]), dtype=np.uint8)
        batch_b = np.zeros(shape=(batch_size, self.imshape[0], self.imshape[1], self.imshape[2]), dtype=np.uint8)

        #get the batch indices, here these correspond to different sample volumes
        batch_indices = np.random.choice(self.sample_indices, size=batch_size, replace=False)

        #get image indices, these correspond to slices
        ### do arange to get img idx array from 0 to 459, then sample from this, then perform operations

        batch_img = np.random.choice(self.nr_img, size=batch_size, replace=False)

        ## Access desired image as follows:
        img_indices = self.aug_factor * batch_img + np.random.choice(self.aug_factor)

        # HDF5 requires indices to be in increasing order
        batch_indices = np.sort(batch_indices)
        img_indices = np.sort(img_indices)

        i = 0
        for idx, img in zip(batch_indices, img_indices):
            batch_a[i,:,:,:] = self.data['A' + '/data_' + str(idx)][img, :, :, :]
            batch_b[i,:,:,:] = self.data['B' + '/data_' + str(idx)][img, :, :, :]
            i += 1

        return batch_a, batch_b
