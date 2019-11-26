
import numpy as np
import utils
import pandas as pd
import warnings
from scipy.stats import norm

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



    def next_batch(self, batch_size):
        """
        Get a single random batch. This implements sampling without replacement (not just on a batch level), this means
        all the data gets sampled eventually.
        """

        if len(self.unused_indices) < batch_size:
            self.unused_indices = self.indices

        batch_indices = np.random.choice(self.unused_indices, batch_size, replace=False)
        self.unused_indices = np.setdiff1d(self.unused_indices, batch_indices)

        # HDF5 requires indices to be in increasing order
        batch_indices = np.sort(batch_indices)

        X_batch = np.float32(self.X[batch_indices, ...])
        y_batch = self.y[batch_indices, ...]

        X_batch, y_batch = self._post_process_batch(X_batch, y_batch)

        return X_batch, y_batch


    def iterate_batches(self, batch_size, shuffle=True):
        """
        Get a range of batches. Use as argument of a for loop like you would normally use
        the range() function.
        """

        if shuffle:
            np.random.shuffle(self.indices)
        N = self.indices.shape[0]

        for b_i in range(0, N, batch_size):

            # if b_i + batch_size > N:
            #     continue

            # HDF5 requires indices to be in increasing order
            batch_indices = np.sort(self.indices[b_i:b_i + batch_size])

            X_batch = np.float32(self.X[batch_indices, ...])
            y_batch = self.y[batch_indices, ...]

            X_batch, y_batch = self._post_process_batch(X_batch, y_batch)

            yield X_batch, y_batch


    def _post_process_batch(self, X_batch, y_batch):

        if self.convert_to_gray:
            X_batch = np.mean(X_batch, axis=-1, keepdims=True)

        if self.do_augmentations:
            X_batch, y_batch = self._augmentation_function(X_batch, y_batch)

        if self.center_images:
            X_batch = utils.center_images(X_batch)

        if self.normalise_images:
            X_batch = utils.normalise_images(X_batch)

        if self.map_to_unity_range:
            X_batch = utils.map_images_to_intensity_range(X_batch, -1, 1, percentiles=0.95)

        if self.add_dummy_dimension:
            X_batch = np.expand_dims(X_batch, axis=-1)

        return X_batch, y_batch

    def get_option(self, name, default):
        return self.augmentation_options[name] if name in self.augmentation_options else default

    def _augmentation_function(self, images, labels):
        '''
        Function for augmentation of minibatches. It will transform a set of images and corresponding labels
        by a number of optional transformations. Each image/mask pair in the minibatch will be seperately transformed
        with random parameters.
        :param images: A numpy array of shape [minibatch, X, Y, (Z), nchannels]
        :param labels: A numpy array containing a corresponding label mask
        :param do_rotations: Rotate the input images by a random angle between -15 and 15 degrees.
        :param do_scaleaug: Do scale augmentation by sampling one length of a square, then cropping and upsampling the image
                            back to the original size.
        :param do_fliplr: Perform random flips with a 50% chance in the left right direction.
        :return: A mini batch of the same size but with transformed images and masks.
        '''

        if images.ndim > 4:
            return self._3d_augmentation(images, labels)
            # raise AssertionError('Augmentation will only work with 2D images')

        try:
            import cv2
        except:
            return False
        else:

            # If segmentation labels also augment them, otherwise don't
            augment_labels = True if labels.ndim > 1 else False

            do_rotations = self.get_option('do_rotations', False)
            do_scaleaug = self.get_option('do_scaleaug', False)
            do_fliplr = self.get_option('do_fliplr', False)
            do_flipud = self.get_option('do_flipud', False)
            do_elasticaug = self.get_option('do_elasticaug', False)
            augment_every_nth = self.get_option('augment_every_nth', 2)  # 2 means augment half of the images
                                                                    # 1 means augment every image

            if do_rotations or do_scaleaug or do_elasticaug:
                nlabels = self.get_option('nlabels', None)
                if not nlabels:
                    raise AssertionError("When doing augmentations with rotations, scaling, or elastic transformations "
                                         "the parameter 'nlabels' must be provided.")


            new_images = []
            new_labels = []
            num_images = images.shape[0]

            for ii in range(num_images):

                img = np.squeeze(images[ii, ...])
                lbl = np.squeeze(labels[ii, ...])

                coin_flip = np.random.randint(augment_every_nth)
                if coin_flip == 0:

                    # ROTATE
                    if do_rotations:

                        angles = self.get_option('rot_degrees', 10.0)
                        random_angle = np.random.uniform(-angles, angles)
                        img = utils.rotate_image(img, random_angle)

                        if augment_labels:
                            if nlabels <= 4:
                                lbl = utils.rotate_image_as_onehot(lbl, random_angle, nlabels=nlabels)
                            else:
                                # If there are more than 4 labels open CV can no longer handle one-hot interpolation
                                lbl = utils.rotate_image(lbl, random_angle, interp=cv2.INTER_NEAREST)

                    # RANDOM CROP SCALE
                    if do_scaleaug:

                        offset = self.get_option('offset', 30)
                        n_x, n_y = img.shape
                        r_y = np.random.random_integers(n_y - offset, n_y)
                        p_x = np.random.random_integers(0, n_x - r_y)
                        p_y = np.random.random_integers(0, n_y - r_y)

                        img = utils.resize_image(img[p_y:(p_y + r_y), p_x:(p_x + r_y)], (n_x, n_y))
                        if augment_labels:
                            if nlabels <= 4:
                                lbl = utils.resize_image_as_onehot(lbl[p_y:(p_y + r_y), p_x:(p_x + r_y)], (n_x, n_y), nlabels=nlabels)
                            else:
                                lbl = utils.resize_image(lbl[p_y:(p_y + r_y), p_x:(p_x + r_y)], (n_x, n_y), interp=cv2.INTER_NEAREST)

                    # RANDOM ELASTIC DEFOMRATIONS (like in U-NET)
                    if do_elasticaug:

                        mu = 0
                        sigma = 10
                        n_x, n_y = img.shape

                        dx = np.random.normal(mu, sigma, 9)
                        dx_mat = np.reshape(dx, (3, 3))
                        dx_img = utils.resize_image(dx_mat, (n_x, n_y), interp=cv2.INTER_CUBIC)

                        dy = np.random.normal(mu, sigma, 9)
                        dy_mat = np.reshape(dy, (3, 3))
                        dy_img = utils.resize_image(dy_mat, (n_x, n_y), interp=cv2.INTER_CUBIC)

                        img = utils.dense_image_warp(img, dx_img, dy_img)

                        if augment_labels:

                            if nlabels <= 4:
                                lbl = utils.dense_image_warp_as_onehot(lbl, dx_img, dy_img, nlabels=nlabels)
                            else:
                                lbl = utils.dense_image_warp(lbl, dx_img, dy_img, interp=cv2.INTER_NEAREST, do_optimisation=False)


                # RANDOM FLIP
                if do_fliplr:
                    coin_flip = np.random.randint(max(2, augment_every_nth))  # Flipping wouldn't make sense if you do it always
                    if coin_flip == 0:
                        img = np.fliplr(img)
                        if augment_labels:
                            lbl = np.fliplr(lbl)

                if do_flipud:
                    coin_flip = np.random.randint(max(2, augment_every_nth))
                    if coin_flip == 0:
                        img = np.flipud(img)
                        if augment_labels:
                            lbl = np.flipud(lbl)

                new_images.append(img[...])
                new_labels.append(lbl[...])

            sampled_image_batch = np.asarray(new_images)
            sampled_label_batch = np.asarray(new_labels)

            return sampled_image_batch, sampled_label_batch

    def _3d_augmentation(self, images, labels):

        try:
            from scipy.ndimage import interpolation, filters
        except:
            return False

        # If segmentation labels also augment them, otherwise don't
        augment_labels = True if len(images.shape) == len(labels.shape) else False

        do_shift = self.get_option('do_shift', False)
        do_rotation = self.get_option('do_rotation', False)
        do_gaussnoise = self.get_option('do_gaussnoise', False)
        do_elastic = self.get_option('do_elastic', False)
        do_flip = self.get_option('do_flip', False)
        augment_every_nth = self.get_option('augment_every_nth', 1)  # 2 means augment half of the images
                                                                     # 1 means augment every image

        new_images = []
        new_labels = []
        num_images = images.shape[0]

        for ii in range(num_images):

            img = np.squeeze(images[ii, ...])
            lbl = np.squeeze(labels[ii, ...])

            coin_flip = np.random.randint(augment_every_nth)
            if coin_flip == 0:

                # SHIFT
                if do_shift:
                    shifts = self.get_option('shift', 10)
                    random_shifts = np.random.uniform(-shifts, shifts, size=3)

                    img = interpolation.shift(img, random_shifts, order=2, cval=0.0)

                    if augment_labels:
                        lbl = interpolation.shift(lbl, random_shifts, order=2, cval=0.0)

                # ROTATION
                if do_rotation:
                    angles = self.get_option('rot_degrees', 30.0)
                    random_angle = np.random.uniform(-angles, angles, size=3)

                    img = interpolation.rotate(img, random_angle[0], order=2, cval=0.0, axes=(0, 1), reshape=False)
                    img = interpolation.rotate(img, random_angle[1], order=2, cval=0.0, axes=(0, 2), reshape=False)
                    img = interpolation.rotate(img, random_angle[2], order=2, cval=0.0, axes=(1, 2), reshape=False)

                    if augment_labels:
                        lbl = interpolation.rotate(lbl, random_angle[0], order=2, cval=0.0, axes=(0, 1), reshape=False)
                        lbl = interpolation.rotate(lbl, random_angle[1], order=2, cval=0.0, axes=(0, 2), reshape=False)
                        lbl = interpolation.rotate(lbl, random_angle[2], order=2, cval=0.0, axes=(1, 2), reshape=False)

                # GAUSSIAN NOISE
                if do_gaussnoise:
                    max_stddev = self.get_option('max_stddev', 0.1)
                    random_std = max_stddev * np.random.rand()
                    noise = np.random.normal(0, random_std, img.shape)
                    img = np.clip(img + noise, 0, 1)  # doesn't make sense to augment label image here

                # ELASTIC DEFORMATION
                if do_elastic:
                    sigma = self.get_option('sigma', 10)
                    max_alpha = self.get_option('max_alpha', 10)
                    alpha = np.random.uniform(0, max_alpha)
                    dx = filters.gaussian_filter((np.random.rand(*img.shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
                    dy = filters.gaussian_filter((np.random.rand(*img.shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
                    dz = filters.gaussian_filter((np.random.rand(*img.shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

                    x, y, z = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1], np.arange(img.shape[2])), indexing='ij')
                    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(z + dz, (-1, 1))

                    img = interpolation.map_coordinates(img, indices, order=1).reshape(img.shape)

                    if augment_labels:
                        lbl = interpolation.map_coordinates(lbl, indices, order=1).reshape(lbl.shape)

            # RANDOM FLIP
            if do_flip:
                coin_flip = np.random.randint(max(2, augment_every_nth))  # Flipping wouldn't make sense if you do it always
                if coin_flip == 0:
                    img = np.flip(img, axis=0)
                    if augment_labels:
                        lbl = np.flip(lbl, axis=0)

                coin_flip = np.random.randint(max(2, augment_every_nth))
                if coin_flip == 0:
                    img = np.flip(img, axis=1)
                    if augment_labels:
                        lbl = np.flip(lbl, axis=1)

                coin_flip = np.random.randint(max(2, augment_every_nth))
                if coin_flip == 0:
                    img = np.flip(img, axis=2)
                    if augment_labels:
                        lbl = np.flip(lbl, axis=2)

            new_images.append(img[...])
            new_labels.append(lbl[...])

        sampled_image_batch = np.asarray(new_images)
        sampled_label_batch = np.asarray(new_labels)

        return sampled_image_batch, sampled_label_batch


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
