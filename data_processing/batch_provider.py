## Strongly inspired by BatchProvider of Christian Baumgartner's implementation in his "discriminative learning toolbox",
import numpy as np
import utils
import pandas as pd
import warnings
from scipy.stats import norm


class BatchProvider_():
    """
    Batch Provider class to provide batches of data for the network
    """
    def __init__(self, A, B, sample_indices, img_indices):
        self.A = A
        self.B = B
        self.sample_indices = sample_indices
        self.img_indices = img_indices

    def iterate_batches(self, batch_size, shuffle=True):

        if shuffle:
            np.random.shuffle(self.indices)
        N = self.indices.shape[0]

        for b_i in range(0, N, batch_size):

            # if b_i + batch_size > N:
            #     continue

            # HDF5 requires indices to be in increasing order
            batch_indices = np.sort(self.indices[b_i:b_i + batch_size])

class BatchProvider():
    """
    This is a helper class to conveniently access mini batches of training, testing and validation data
    """

    def __init__(self, X, y, indices, img_indices, normalise_images=False, map_to_unity_range=False, add_dummy_dimension=True, **kwargs):
        # indices don't always cover all of X and Y (e.g. in the case of val set)

        self.X = X
        self.y = y
        self.indices = indices
        self.img_indices = img_indices
        self.unused_indices = indices.copy()
        self.normalise_images = normalise_images
        self.add_dummy_dimension = add_dummy_dimension
        self.map_to_unity_range = map_to_unity_range
        self.center_images = kwargs.get('center_images', False)
        self.unity_percentile = kwargs.get('unity_percentile', 0.95)

        self.do_augmentations = kwargs.get('do_augmentations', False)
        self.augmentation_options = kwargs.get('augmentation_options', None)

        self.convert_to_gray = kwargs.get('convert_to_gray', False)


class AugDataBatchProvider(BatchProvider):
    def __init__(self, X, y, lab_name, feat, feat_names, img_infl_feat, standalone_feat, indices, img_indices, impute_k, exp_config,
                 nr_dup, aux_lab_name=tuple(), augment=False, ref_table=None, balance_sampling=False, standardize_features=True, include_table=True, **kwargs):
        super(AugDataBatchProvider, self).__init__(X, y, indices, img_indices, **kwargs)
        self.lab_name = lab_name
        self.aux_lab_name = list(aux_lab_name)
        self.feat = feat
        self.img_infl_feat = img_infl_feat
        self.standalone_feat = standalone_feat
        self.feat_names = feat_names
        self.nr_dup = nr_dup
        self.augment = augment
        self.impute_k = impute_k
        self.balance_sampling = balance_sampling
        self.standardize_features = standardize_features
        self.include_table = include_table

        self.load_table(ref_table)
        self.exp_config = exp_config

        if balance_sampling:
            self.weights = self.get_balanced_sampling_weights()
        else:
            self.weights = np.ones(len(self.indices))

    def iterate_batches(self, batch_size, shuffle=True, image_indic_arg=None):
        """
        Get a range of batches. Use as argument of a for loop like you would normally use
        the range() function.
        """

        if self.balance_sampling:
            warnings.warn("balance_sampling does not work for iterate_batches, sampling is imbalanced!")

        if shuffle:
            np.random.shuffle(self.indices)
        N = self.indices.shape[0]

        for b_i in range(0, N, batch_size):

            # if b_i + batch_size > N:
            #     continue

            # HDF5 requires indices to be in increasing order
            batch_indices = np.sort(self.indices[b_i:b_i + batch_size])


            # generate a bool array of the same shape as batch_indices, indicating where images are missing in batch (0)
            img_list = []
            for i in range(len(batch_indices)):
                if batch_indices[i] in self.img_indices:
                    img_list.append(1)
                else:
                    img_list.append(0)

            img_nan_indices = np.asarray(img_list).reshape((len(img_list), 1))


            if image_indic_arg:
                image_indic = [self.nr_dup * image_indic_arg[b_i]]
            else:
                image_indic = []
                if self.augment:
                    # choose a random augmented version of that image
                    # image_indic = [self.nr_dup * el + np.random.randint(self.nr_dup) for el in batch_indices]
                    for i in range(len(batch_indices)):
                        if batch_indices[i] in self.img_indices:
                            #TODO: had to add ugly -term to random selection since there are less images than patients
                            # otherwise problems when there are too many patients without image in batch (due to hdf5 index troubles)
                            image_indic.append(self.nr_dup * batch_indices[i] + np.random.randint(self.nr_dup - batch_size))
                        else:
                            # choose dummy image which is certainly present
                            if len(image_indic) is not 0:
                                image_indic.append(image_indic[i - 1] + 1)
                            else:
                                image_indic.append(self.img_indices[0])
                ## else part might not work with prob. model in current setup --> hdf5 is a pain to work with, might have to make it nicer in the future
                else:
                    # image_indic = [self.nr_dup * el for el in batch_indices]
                    for i in range(len(batch_indices)):
                        if batch_indices[i] in self.img_indices:
                            image_indic.append(self.nr_dup * batch_indices[i])
                        else:
                            # choose dummy image which is certainly present
                            if len(image_indic) is not 0:
                                image_indic.append(image_indic[i - 1] + 1)
                            else:
                                image_indic.append(self.img_indices[0])

            X_batch = np.float32(self.X[image_indic, ...])
            y_batch = [self.y[self.lab_name][batch_indices, ...]]
            # for el in self.aux_lab_name:
            #     y_batch.append(self.table[el].loc[batch_indices])
            y_batch = np.swapaxes(y_batch, 0, 1)

            table_batch = self.table[self.feat_names].loc[batch_indices]
            prior_img_batch = self.table[self.img_infl_feat].loc[batch_indices]
            prior_nonimg_batch = self.table[self.standalone_feat].loc[batch_indices]
            J_batch = self.table[self.aux_lab_name].loc[batch_indices]

            X_batch, y_batch = self._post_process_batch(X_batch, y_batch)

            if self.do_augmentations:
                do_tablenoise = self.get_option('do_tablenoise', False)
                if do_tablenoise:
                    table_batch = self.add_table_noise(table_batch)
                    prior_img_batch = self.add_table_noise(prior_img_batch)
                    prior_nonimg_batch = self.add_table_noise(prior_nonimg_batch)
                    J_batch = self.add_table_noise(J_batch)

            ## add priors, really ugly way of doing it but don't have time to think about sophisticated way
            for i in range(table_batch.shape[1]):
                value_list = []
                if i < len(self.exp_config.cont_feat):
                    prob = norm.pdf(table_batch.values[:, i])
                    value_list.append(prob)
                    values = np.asarray(value_list).reshape((table_batch.shape[0], 1))
                    if i == 0:
                        values_tot = values
                    else:
                        values_tot = np.c_[values_tot, values]
                else:
                    prob = self.exp_config.cat_feat_prob[i - len(self.exp_config.cont_feat)]
                    values_cat = np.where(table_batch.values[:, i] == 1, prob, 1 - prob)
                    values_tot = np.c_[values_tot, values_cat]

            priors = values_tot


            if self.include_table:
                yield X_batch, y_batch, table_batch, J_batch, prior_img_batch, prior_nonimg_batch, img_nan_indices, priors
            else:
                yield X_batch, y_batch, img_nan_indices

    def next_batch(self, batch_size):
        """
        Get a single random batch. This implements random sampling with replacement.
        """

        batch_indices = np.random.choice(self.indices, size=batch_size, replace=False, p=self.weights)

        # HDF5 requires indices to be in increasing order
        batch_indices = np.sort(batch_indices)

        # generate a bool array of the same shape as batch_indices, indicating where images are missing in batch (0)
        img_list = []
        for i in range(len(batch_indices)):
            if batch_indices[i] in self.img_indices:
                img_list.append(1)
            else:
                img_list.append(0)

        img_nan_indices = np.asarray(img_list).reshape((len(img_list), 1))

        image_indic = []
        if self.augment:
            # choose a random augmented version of that image
            # image_indic = [self.nr_dup * el + np.random.randint(self.nr_dup) for el in batch_indices]
            for i in range(len(batch_indices)):
                if batch_indices[i] in self.img_indices:
                    #added a -1 here so we can choose the last image of the previous image_indic entry for the new one
                    #in case we get to the else condition (hdf5 needs indices in increasing order)
                    ##TODO: Change this here not to have the -10, just for testing purposes whether program works
                    image_indic.append(self.nr_dup * batch_indices[i] + np.random.randint(self.nr_dup - batch_size))
                else:
                    # choose dummy image which is certainly present
                    if len(image_indic) is not 0:
                        image_indic.append(image_indic[i - 1] + 1)
                    else:
                        image_indic.append(self.img_indices[0])
        ## else part might not work with prob. model in current setup --> hdf5 is a pain to work with, might have to make it nicer in the future
        else:
            # image_indic = [self.nr_dup * el for el in batch_indices]
            for i in range(len(batch_indices)):
                if batch_indices[i] in self.img_indices:
                    image_indic.append(self.nr_dup * batch_indices[i])
                else:
                    # choose dummy image which is certainly present
                    if len(image_indic) is not 0:
                        image_indic.append(image_indic[i - 1] + 1)
                    else:
                        image_indic.append(self.img_indices[0])

        X_batch = np.float32(self.X[image_indic, ...])
        y_batch = [self.y[self.lab_name][batch_indices, ...]]
        # for el in self.aux_lab_name:
        #     y_batch.append(self.table[el].loc[batch_indices])
        y_batch = np.swapaxes(y_batch, 0, 1)

        table_batch = self.table[self.feat_names].loc[batch_indices]
        prior_img_batch = self.table[self.img_infl_feat].loc[batch_indices]
        prior_nonimg_batch = self.table[self.standalone_feat].loc[batch_indices]
        J_batch = self.table[self.aux_lab_name].loc[batch_indices]

        X_batch, y_batch = self._post_process_batch(X_batch, y_batch)

        if self.do_augmentations:
           do_tablenoise = self.get_option('do_tablenoise', False)
           if do_tablenoise:
               table_batch = self.add_table_noise(table_batch)
               prior_img_batch = self.add_table_noise(prior_img_batch)
               prior_nonimg_batch = self.add_table_noise(prior_nonimg_batch)
               J_batch = self.add_table_noise(J_batch)

        ## add priors, really ugly way of doing it but don't have time to think about sophisticated way
        for i in range(table_batch.shape[1]):
            value_list = []
            if i < len(self.exp_config.cont_feat):
                prob = norm.pdf(table_batch.values[:, i])
                value_list.append(prob)
                values = np.asarray(value_list).reshape((table_batch.shape[0], 1))
                if i == 0:
                    values_tot = values
                else:
                    values_tot = np.c_[values_tot, values]
            else:
                prob = self.exp_config.cat_feat_prob[i - len(self.exp_config.cont_feat)]
                values_cat = np.where(table_batch.values[:, i] == 1, prob, 1 - prob)
                values_tot = np.c_[values_tot, values_cat]

        priors = values_tot

        if self.include_table:
            return X_batch, y_batch, table_batch, J_batch, prior_img_batch, prior_nonimg_batch, img_nan_indices, priors
        else:
            return X_batch, y_batch, img_nan_indices
