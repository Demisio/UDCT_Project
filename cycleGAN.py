from __future__ import division, print_function, unicode_literals


import tensorflow as tf

import h5py
import numpy as np
import time
import os
import logging
import sys
import cv2
from sklearn.metrics import f1_score, roc_auc_score
from scipy.stats import pearsonr
from collections import deque
import matplotlib.pyplot as plt

sys.path.append('./Discriminator')
sys.path.append('./Generator')
sys.path.append('./Utilities/')
from Generator import Res_Gen
from Discriminator import PatchGAN34
from Discriminator import PatchGAN70
from Discriminator import PatchGAN142
from Discriminator import MultiPatch
from Discriminator import HisDis
from Utilities import Utilities

from tfwrapper import utils as tf_utils
from tfwrapper.losses import mean_dice
from data_processing import heart_data, batch_provider
from utils import seedKeeper


class Model:
    """
    ToDo
    -) save()      - Save the current model parameter
    -) create()    - Create the model layers
    -) init()      - Initialize the model (load model if exists)
    -) load()      - Load the parameters from the file
    -) ToDo
    
    Only the following functions should be called from outside:
    -) ToDo
    -) constructor
    """
    
    def __init__(self,
                 mod_name,
                 data_file,
                 buffer_size=32,
                 architecture='Res6',
                 lambda_h=10.,\
                 lambda_c=10.,\
                 dis_noise=0.25,\
                 deconv='transpose',\
                 patchgan='Patch70',\
                 verbose=False,\
                 gen_only=False,
                 log_name='filler',
                 fold=1,
                 checkpoints=1):
        """
        Create a Model (init). It will check, if a model with such a name has already been saved. If so, the model is being 
        loaded. Otherwise, a new model with this name will be created. It will only be saved, if the save function is being 
        called. The describtion of every parameter is given in the code below.
        
        INPUT: mod_name      - This is the name of the model. It is mainly used to establish the place, where the model is being 
                               saved.
               data_file     - hdf5 file that contains the dataset
               imsize        - The dimension of the input images
                              
        OUTPUT:             - The model
        """
        
        self.mod_name            = mod_name                               # Model name (see above)
        
        self.data_file           = data_file                              # hdf5 data file

        self.fold                = fold

        self.checkpoints         = checkpoints                            #whether to do checkpoints or not (default = True)

        self.seed = 42
        np.random.seed(self.seed)


        ## experiment configuration, non-automatic so far
        self.data = heart_data.heart_data(self.data_file, self.fold)

        ## different sets
        self.train_data = self.data.train
        self.validation_data = self.data.validation
        self.test_data = self.data.test

        ## other model params
        self.a_chan = self.data.a_chan
        self.b_chan = self.data.b_chan
        self.imsize = self.data.imsize
        self.a_size = self.data.a_size
        self.b_size = self.data.b_size


        # Reset all current saved tf stuff
        tf.reset_default_graph()
        
        self.architecture        = architecture
        self.lambda_h            = lambda_h
        self.lambda_c            = lambda_c
        self.dis_noise_0         = dis_noise
        self.deconv              = deconv
        self.patchgan            = patchgan
        self.verbose             = verbose
        self.gen_only            = gen_only  # If true, only the generator are used (and loaded)
        
        # Create the model that is built out of two discriminators and a generator
        self.create()
        
        # Image buffer
        self.buffer_size         = buffer_size
        self.temp_b_s            = 0.
        self.buffer_real_a       = np.zeros([self.buffer_size, self.imsize, self.imsize, self.a_chan])
        self.buffer_real_b       = np.zeros([self.buffer_size, self.imsize, self.imsize, self.b_chan])
        self.buffer_fake_a       = np.zeros([self.buffer_size, self.imsize, self.imsize, self.a_chan])
        self.buffer_fake_b       = np.zeros([self.buffer_size, self.imsize, self.imsize, self.b_chan])
        
        # # Create the model saver
        # with self.graph.as_default():
        #     if not self.gen_only:
        #         self.saver    = tf.train.Saver()
        #     else:
        #         self.saver    = tf.train.Saver(var_list=self.list_gen)

        with self.graph.as_default():
            self.saver = tf.train.Saver(max_to_keep=2)
            self.saver_best_dice = tf.train.Saver(max_to_keep=2)


        self.log_name = log_name
        self.fold_name = 'fold_' + str(self.fold)




    
    def create(self):
        """
        Create the model. ToDo
        """
        # Create a graph and add all layers
        self.graph = tf.Graph()

        with self.graph.as_default():

            tf.set_random_seed(self.seed)

            # Define variable learning rate and dis_noise
            self.relative_lr    = tf.placeholder_with_default([1.],[1],name="relative_lr")
            self.relative_lr    = self.relative_lr[0]
            
            self.rel_dis_noise  = tf.placeholder_with_default([1.],[1],name="rel_dis_noise")
            self.rel_dis_noise  = self.rel_dis_noise[0]
            self.dis_noise      = self.rel_dis_noise * self.dis_noise_0
            
            
            # Create the generator and discriminator
            if self.architecture == 'Res6':
                                gen_dim =    [64,   128,256,   256,256,256,256,256,256,   128,64     ]
                                kernel_size =[7,    3,3,       3,3,3,3,3,3,               3,3,      7]
            elif self.architecture == 'Res9':
                                gen_dim=    [64,   128,256,   256,256,256,256,256,256,256,256,256,   128,64    ]
                                kernel_size=[7,    3,3,       3,3,3,3,3,3,3,3,3,                     3,3,     7]
            else:
                                print('Unknown generator architecture')
                                return None
            
            self.genA           = Res_Gen.ResGen('BtoA',self.a_chan,gen_dim=gen_dim,kernel_size=kernel_size,deconv=self.deconv,verbose=self.verbose)
            self.genB           = Res_Gen.ResGen('AtoB',self.b_chan,gen_dim=gen_dim,kernel_size=kernel_size,deconv=self.deconv,verbose=self.verbose)
            
            if self.patchgan == 'Patch34':
                self.disA       = PatchGAN34.PatchGAN34('A',noise=self.dis_noise)
                self.disB       = PatchGAN34.PatchGAN34('B',noise=self.dis_noise)
            elif self.patchgan == 'Patch70':
                self.disA       = PatchGAN70.PatchGAN70('A',noise=self.dis_noise)
                self.disB       = PatchGAN70.PatchGAN70('B',noise=self.dis_noise)
            elif self.patchgan == 'Patch142':
                self.disA       = PatchGAN142.PatchGAN142('A',noise=self.dis_noise)
                self.disB       = PatchGAN142.PatchGAN142('B',noise=self.dis_noise)
            elif self.patchgan == 'MultiPatch':
                self.disA       = MultiPatch.MultiPatch('A',noise=self.dis_noise)
                self.disB       = MultiPatch.MultiPatch('B',noise=self.dis_noise)
            else:
                print('Unknown Patch discriminator type')
                return None
            
            self.disA_His   = HisDis.HisDis('A',noise=self.dis_noise,keep_prob=1.)
            self.disB_His   = HisDis.HisDis('B',noise=self.dis_noise,keep_prob=1.)
        
            # Create a placeholder for the input data
            self.A           = tf.placeholder(tf.float32,[None, None, None, self.a_chan],name="a")
            self.B           = tf.placeholder(tf.float32,[None, None, None, self.b_chan],name="b")
            
            if self.verbose:
                print('Size A: ' +str(self.a_chan)) # Often 1 --> Real
                print('Size B: ' +str(self.b_chan)) # Often 3 --> Syn
            
            # Create cycleGAN                
            
            self.fake_A      = self.genA.create(self.B,False)
            self.fake_B      = self.genB.create(self.A,False)
            
            
            
            # Define the histogram loss
            t_A             = tf.transpose(tf.reshape(self.A,[-1, self.a_chan]),[1,0])
            t_B             = tf.transpose(tf.reshape(self.B,[-1, self.b_chan]),[1,0])
            t_fake_A        = tf.transpose(tf.reshape(self.fake_A,[-1, self.a_chan]),[1,0])
            t_fake_B        = tf.transpose(tf.reshape(self.fake_B,[-1, self.b_chan]),[1,0])

            self.s_A,_      = tf.nn.top_k(t_A,tf.shape(t_A)[1])
            self.s_B,_      = tf.nn.top_k(t_B,tf.shape(t_B)[1])
            self.s_fake_A,_ = tf.nn.top_k(t_fake_A,tf.shape(t_fake_A)[1])
            self.s_fake_B,_ = tf.nn.top_k(t_fake_B,tf.shape(t_fake_B)[1])
            
            self.m_A        = tf.reshape(tf.reduce_mean(tf.reshape(self.s_A,[self.a_chan, self.imsize, -1]),axis=2),[1, -1])
            self.m_B        = tf.reshape(tf.reduce_mean(tf.reshape(self.s_B,[self.b_chan, self.imsize, -1]),axis=2),[1, -1])
            self.m_fake_A   = tf.reshape(tf.reduce_mean(tf.reshape(self.s_fake_A,[self.a_chan, self.imsize, -1]),axis=2),[1, -1])
            self.m_fake_B   = tf.reshape(tf.reduce_mean(tf.reshape(self.s_fake_B,[self.b_chan, self.imsize, -1]),axis=2),[1, -1])
            
            # Define generator loss functions
            self.lambda_c    = tf.placeholder_with_default([self.lambda_c],[1],name="lambda_c")
            self.lambda_c    = self.lambda_c[0]
            self.lambda_h    = tf.placeholder_with_default([self.lambda_h],[1],name="lambda_h")
            self.lambda_h    = self.lambda_h[0]
            
            self.dis_real_A  = self.disA.create(self.A,False)
            self.dis_real_Ah = self.disA_His.create(self.m_A,False)
            self.dis_real_B  = self.disB.create(self.B,False)
            self.dis_real_Bh = self.disB_His.create(self.m_B,False)
            self.dis_fake_A  = self.disA.create(self.fake_A,True)
            self.dis_fake_Ah = self.disA_His.create(self.m_fake_A,True)
            self.dis_fake_B  = self.disB.create(self.fake_B,True)
            self.dis_fake_Bh = self.disB_His.create(self.m_fake_B,True)
            
            self.cyc_A       = self.genA.create(self.fake_B,True)
            self.cyc_B       = self.genB.create(self.fake_A,True)
            
            
            # Define cycle loss (eq. 2)
            self.loss_cyc_A  = tf.reduce_mean(tf.abs(self.cyc_A-self.A))
            self.loss_cyc_B  = tf.reduce_mean(tf.abs(self.cyc_B-self.B))
            
            self.loss_cyc    = self.loss_cyc_A + self.loss_cyc_B
            
            # Define discriminator losses (eq. 1)
            self.loss_dis_A  = (tf.reduce_mean(tf.square(self.dis_real_A)) +\
                                tf.reduce_mean(tf.square(1-self.dis_fake_A)))*0.5 +\
                               (tf.reduce_mean(tf.square(self.dis_real_Ah)) +\
                                tf.reduce_mean(tf.square(1-self.dis_fake_Ah)))*0.5*self.lambda_h
                                
                               
            self.loss_dis_B  = (tf.reduce_mean(tf.square(self.dis_real_B)) +\
                                tf.reduce_mean(tf.square(1-self.dis_fake_B)))*0.5 +\
                               (tf.reduce_mean(tf.square(self.dis_real_Bh)) +\
                                tf.reduce_mean(tf.square(1-self.dis_fake_Bh)))*0.5*self.lambda_h
            
            self.loss_gen_A  = tf.reduce_mean(tf.square(self.dis_fake_A)) +\
                               self.lambda_h * tf.reduce_mean(tf.square(self.dis_fake_Ah)) +\
                               self.lambda_c * self.loss_cyc/2.
            self.loss_gen_B  = tf.reduce_mean(tf.square(self.dis_fake_B)) +\
                               self.lambda_h * tf.reduce_mean(tf.square(self.dis_fake_Bh)) +\
                               self.lambda_c * self.loss_cyc/2.


            # Optimizer for Gen
            self.list_gen        = []
            for var in tf.trainable_variables():
                if 'gen' in str(var):
                    self.list_gen.append(var)
            optimizer_gen   = tf.train.AdamOptimizer(learning_rate=self.relative_lr*0.0002,beta1=0.5)
            self.opt_gen    = optimizer_gen.minimize(self.loss_gen_A+self.loss_gen_B,var_list=self.list_gen)
            
            # Optimizer for Dis
            self.list_dis      = []
            for var in tf.trainable_variables():
                if 'dis' in str(var):
                    self.list_dis.append(var)
            optimizer_dis = tf.train.AdamOptimizer(learning_rate=self.relative_lr*0.0002,beta1=0.5)
            self.opt_dis  = optimizer_dis.minimize(self.loss_dis_A + self.loss_dis_B,var_list=self.list_dis)

            # Initialize Dice loss in TF
            # self.B_mask = tf.placeholder(tf.int32,[None, None, None, self.b_chan],name="B_mask")
            # self.fake_B_mask = tf.placeholder(tf.int32,[None, None, None, self.b_chan],name="fake_B_mask")
            self.n_labels = tf.placeholder(tf.int32, shape=[], name='n_labels')

            self.get_mean_dice = mean_dice(prediction=self.fake_B,
                                           ground_truth=self.B,
                                           nr_labels=self.n_labels,
                                           sum_over_batches=True,
                                           partial_dice=False)

            # Settings to optimize GPU memory usage
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.gpu_options.per_process_gpu_memory_fraction = 1.0

            # Session & variables
            self.sess = tf.Session(graph=self.graph, config=config)
            self.init = tf.global_variables_initializer()



    def train(self, batch_size=32,lambda_c=0.,lambda_h=0.,n_epochs=0,save=True,syn_noise=0.,real_noise=0.):



        self.batch_size = batch_size
        # Sort out proper logging, also add checkpoints & continue from there if necessary
        self._setup_log_dir_and_continue_mode()
        # Create tensorboard summaries (AUC values, losses, accuracy etc.)
        self._make_tensorboard_summaries()

        self.sess.run(self.init)

        loss_gen_A_list = []
        loss_gen_B_list = []
        loss_dis_A_list = []
        loss_dis_B_list = []

        # How many trainable params in model?
        # count_params = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
        # np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.all_variables()])
        # print('Number of trainable Parameters in model: ' + self.sess.run(str(count_params)))
        self.show_params()

        real_start_time = time.time()

        # initialise dice_score & deque for consideration of last couple iterations
        best_dice_score = 0
        best_dice_deque = deque([0] * 5, maxlen=5)
        best_dice_epoch = 0

        best_corr_score = 0
        best_corr_epoch = 0
        corr_best_dice = 0

        if self.continue_run:
            self.saver.restore(self.sess, self.init_checkpoint_path)

        print('INFO:   Starting Training')
        for epoch in range(1, n_epochs+1):

            print('')
            print('Epoch : %d' % (epoch))
            print('')

            num_samples = self.data.nr_images
            num_iterations = num_samples // batch_size


            if self.verbose:
                print('lambda_c: ' + str(lambda_c))
                print('lambda_h: ' + str(lambda_h))

            start_time = time.time()
            # print('Start time of Epoch %d : %.3f' % (epoch, start_time))



            vec_lcA     = []
            vec_lcB     = []

            vec_ldrA    = []
            vec_ldrAh   = []
            vec_ldrB    = []
            vec_ldrBh   = []
            vec_ldfA    = []
            vec_ldfAh   = []
            vec_ldfB    = []
            vec_ldfBh   = []

            vec_l_dis_A = []
            vec_l_dis_B = []
            vec_l_gen_A = []
            vec_l_gen_B = []

            rel_lr = 1.
            if epoch > 100:
                rel_lr = 2. - epoch/100.

            if epoch < 100:
                rel_noise = 0.9**epoch
            else:
                rel_noise = 0.

            for iteration in range(int(num_iterations)):
                images_a, images_b   = self.train_data.next_batch(self.batch_size)

                if images_a.dtype=='uint8':
                    images_a=images_a/float(2**8-1)
                elif images_a.dtype=='uint16':
                    images_a=images_a/float(2**16-1)
                else:
                    raise ValueError('Dataset A is not int8 or int16')
                if images_b.dtype=='uint8':
                    images_b=images_b/float(2**8-1)
                elif images_b.dtype=='uint16':
                    images_b=images_b/float(2**16-1)
                else:
                    raise ValueError('Dataset B is not int8 or int16')

                # images_a  += np.random.randn(*images_a.shape)*real_noise
                # images_b  += np.random.randn(*images_b.shape)*syn_noise

                _, l_gen_A, im_fake_A, l_gen_B, im_fake_B, cyc_A, cyc_B, sA, sB, sfA, sfB, lcA, lcB = self.sess.run([self.opt_gen,\
                                                                        self.loss_gen_A,\
                                                                        self.fake_A,\
                                                                        self.loss_gen_B,\
                                                                        self.fake_B,\
                                                                        self.cyc_A,\
                                                                        self.cyc_B,\
                                                                        self.s_A,self.s_B,self.s_fake_A,self.s_fake_B,\
                                                                        self.loss_cyc_A,\
                                                                        self.loss_cyc_B],\
                                                feed_dict={self.A: images_a,\
                                                           self.B: images_b,\
                                                           self.lambda_c: lambda_c,\
                                                           self.lambda_h: lambda_h,\
                                                           self.relative_lr: rel_lr,\
                                                           self.rel_dis_noise: rel_noise})

                if self.temp_b_s >= self.buffer_size:
                    rand_vec_a = np.random.permutation(self.buffer_size)[:batch_size]
                    rand_vec_b = np.random.permutation(self.buffer_size)[:batch_size]

                    self.buffer_real_a[rand_vec_a,...] = images_a
                    self.buffer_real_b[rand_vec_b,...] = images_b
                    self.buffer_fake_a[rand_vec_a,...] = im_fake_A
                    self.buffer_fake_b[rand_vec_b,...] = im_fake_B
                else:
                    low                                = int(self.temp_b_s)
                    high                               = int(min(self.temp_b_s + batch_size,self.buffer_size))
                    self.temp_b_s                      = high

                    self.buffer_real_a[low:high,...]   = images_a[:(high-low),...]
                    self.buffer_real_b[low:high,...]   = images_b[:(high-low),...]
                    self.buffer_fake_a[low:high,...]   = im_fake_A[:(high-low),...]
                    self.buffer_fake_b[low:high,...]   = im_fake_B[:(high-low),...]

                # Create dataset out of buffer and gen images to train dis
                dis_real_a                         = np.copy(images_a)
                dis_real_b                         = np.copy(images_b)
                dis_fake_a                         = np.copy(im_fake_A)
                dis_fake_b                         = np.copy(im_fake_B)

                half_b_s                           = int(batch_size/2)
                rand_vec_a                         = np.random.permutation(self.temp_b_s)[:half_b_s]
                rand_vec_b                         = np.random.permutation(self.temp_b_s)[:half_b_s]
                dis_real_a[:half_b_s,...]          =  self.buffer_real_a[rand_vec_a,...]
                dis_fake_a[:half_b_s,...]          =  self.buffer_fake_a[rand_vec_a,...]
                dis_real_b[:half_b_s,...]          =  self.buffer_real_b[rand_vec_b,...]
                dis_fake_b[:half_b_s,...]          =  self.buffer_fake_b[rand_vec_b,...]

                _, l_dis_A, l_dis_B, \
                ldrA,ldrAh,ldfA,ldfAh,\
                ldrB,ldrBh,ldfB,ldfBh = self.sess.run([\
                                                self.opt_dis,
                                                self.loss_dis_A,
                                                self.loss_dis_B,
                                                self.dis_real_A,
                                                self.dis_real_Ah,
                                                self.dis_fake_A,
                                                self.dis_fake_Ah,
                                                self.dis_real_B,
                                                self.dis_real_Bh,
                                                self.dis_fake_B,
                                                self.dis_fake_Bh],feed_dict={self.A: dis_real_a,\
                                                                             self.B: dis_real_b,\
                                                                             self.fake_A: dis_fake_a,\
                                                                             self.fake_B: dis_fake_b,\
                                                                             self.lambda_c: lambda_c,\
                                                                             self.lambda_h: lambda_h,\
                                                                             self.relative_lr: rel_lr,\
                                                                             self.rel_dis_noise: rel_noise})

                vec_l_dis_A.append(l_dis_A)
                vec_l_dis_B.append(l_dis_B)
                vec_l_gen_A.append(l_gen_A)
                vec_l_gen_B.append(l_gen_B)

                vec_lcA.append(lcA)
                vec_lcB.append(lcB)

                vec_ldrA.append(ldrA)
                vec_ldrAh.append(ldrAh)
                vec_ldrB.append(ldrB)
                vec_ldrBh.append(ldrBh)
                vec_ldfA.append(ldfA)
                vec_ldfAh.append(ldfAh)
                vec_ldfB.append(ldfB)
                vec_ldfBh.append(ldfBh)

                if np.shape(images_b)[-1]==4:

                    images_b=np.vstack((images_b[0,:,:,0:3],np.tile(images_b[0,:,:,3].reshape(320,320,1),[1,1,3])))
                    im_fake_B=np.vstack((im_fake_B[0,:,:,0:3],np.tile(im_fake_B[0,:,:,3].reshape(320,320,1),[1,1,3])))
                    cyc_B=np.vstack((cyc_B[0,:,:,0:3],np.tile(cyc_B[0,:,:,3].reshape(320,320,1),[1,1,3])))
                    images_b=images_b[np.newaxis,:,:,:]
                    im_fake_B=im_fake_B[np.newaxis,:,:,:]
                    cyc_B=cyc_B[np.newaxis,:,:,:]

                if iteration%5==0:
                    self.sneak_peak=Utilities.produce_tiled_images(images_a,images_b,im_fake_A, im_fake_B,cyc_A,cyc_B)

                    cv2.imshow("",self.sneak_peak[:,:,[2,1,0]])
                    cv2.waitKey(1)

                if iteration%20==0:
                    print("\rTrain: {}/{} ({:.1f}%)".format(iteration+1, num_iterations,(iteration) * 100 / (num_iterations-1)) + \
                          "          Loss_dis_A={:.4f},   Loss_dis_B={:.4f}".format(np.mean(vec_l_dis_A),np.mean(vec_l_dis_B)) + \
                          ",   Loss_gen_A={:.4f},   Loss_gen_B={:.4f}".format(np.mean(vec_l_gen_A),np.mean(vec_l_gen_B))\
                              ,end="        ")

            # Validation / Evaluation
            # TODO: checkpoints here
            # if epoch % 5 == 0:
            print('')
            print('--- Evaluation of Training Data ---')

            train_mean_dice, tr_frac_list_b, tr_frac_list_fake_b, tr_corr, sk_train_dice = self.do_validation(self.train_data.iterate_batches)

            tr_b_meanfrac = np.mean(np.asarray(tr_frac_list_b))
            tr_b_varfrac = np.var(np.asarray(tr_frac_list_b))
            tr_fk_b_meanfrac = np.mean(np.asarray(tr_frac_list_fake_b))
            tr_fk_b_varfrac = np.var(np.asarray(tr_frac_list_fake_b))

            print('Training Dice Score: {}'.format(train_mean_dice))
            print('Training SKLearn Dice Score: {}'.format(sk_train_dice))
            print('Training Pearson Correlation of Collagen Fraction: {}'.format(tr_corr))
            print('Training Collagen Fraction Real Image - Mean: {} / Variance: {}'.format(tr_b_meanfrac,
                                                                                             tr_b_varfrac))
            print('Training Collagen Fraction Fake Image - Mean: {} / Variance: {}'.format(tr_fk_b_meanfrac,
                                                                                             tr_fk_b_varfrac))
            print('')
            print('--- Evaluation of Validation Data ---')

            val_mean_dice, val_frac_list_b, val_frac_list_fake_b, val_corr, sk_val_dice = self.do_validation(self.validation_data.iterate_batches)

            val_b_meanfrac = np.mean(np.asarray(val_frac_list_b))
            val_b_varfrac = np.var(np.asarray(val_frac_list_b))
            val_fk_b_meanfrac = np.mean(np.asarray(val_frac_list_fake_b))
            val_fk_b_varfrac = np.var(np.asarray(val_frac_list_fake_b))

            print('Validation Dice Score: {}'.format(val_mean_dice))
            print('Validation SKLearn Dice Score: {}'.format(sk_val_dice))
            print('Validation Pearson Correlation of Collagen Fraction: {}'.format(val_corr))
            print('Validation Collagen Fraction Real Image - Mean: {} / Variance: {}'.format(val_b_meanfrac,
                                                                                             val_b_varfrac))
            print('Validation Collagen Fraction Fake Image - Mean: {} / Variance: {}'.format(val_fk_b_meanfrac,
                                                                                             val_fk_b_varfrac))

            # Plots to check the collagen fractions in slices
            # self.corr_plots(tr_frac_list_b, tr_frac_list_fake_b, val_frac_list_b, val_frac_list_fake_b)

            # Use deque to consider the last 3 validation results to avoid sudden jumps in model performance
            best_dice_deque.append(val_mean_dice)
            smoothed_best_dice = np.mean(best_dice_deque)

            ## TODO: Add condition to only include the collagen fraction when working with heart data
            if smoothed_best_dice >= best_dice_score:
                best_dice_score = smoothed_best_dice
                best_dice_epoch = epoch
                corr_best_dice = val_corr # get the dice score at this epoch to see if it corresponds
                best_file = os.path.join(self.log_dir, 'model_best_dice.ckpt')
                self.saver_best_dice.save(self.sess, best_file, global_step=epoch)
                print('INFO:  Found new best Dice score on validation set (smoothed)! - %f -  Saving model_best_dice.ckpt' % smoothed_best_dice)
                print('Corresponding Pearson Corr. is: {}'.format(corr_best_dice))
                print('INFO:  Epoch = {}'.format(best_dice_epoch))

            if val_corr >= best_corr_score:
                best_corr_score = val_corr
                best_corr_epoch = epoch
                print('INFO:  Found new best Pearson corr. on validation set (non-smoothed): {}'.format(best_corr_score))
                print('INFO:  Epoch = {}'.format(best_corr_epoch))

            train_summary_metrics = self.sess.run(self.train_summary,
                                                      feed_dict={self.train_mean_dice_summary_: train_mean_dice,
                                                      self.train_corr_summary_: tr_corr,
                                                      self.train_sk_dice_summary_: sk_train_dice,
                                                      self.train_coll_frac_real_summary_: np.asarray(tr_frac_list_b),
                                                      self.train_coll_frac_fake_summary_: np.asarray(tr_frac_list_fake_b)})

            val_summary_metrics = self.sess.run(self.val_summary,
                                                feed_dict={self.val_mean_dice_summary_: val_mean_dice,
                                                self.val_corr_summary_: val_corr,
                                                self.val_sk_dice_summary_: sk_val_dice,
                                                self.val_coll_frac_real_summary_: np.asarray(val_frac_list_b),
                                                self.val_coll_frac_fake_summary_: np.asarray(val_frac_list_fake_b)})



            train_summary_dis = self.sess.run(self.train_loss_dis,
                                              feed_dict = {self.train_loss_dis_A_summary_ : np.mean(vec_l_dis_A),
                                                           self.train_loss_dis_B_summary_ : np.mean(vec_l_dis_B)})

            train_summary_gen = self.sess.run(self.train_loss_gen,
                                              feed_dict = {self.train_loss_gen_A_summary_: np.mean(vec_l_gen_A),
                                                           self.train_loss_gen_B_summary_: np.mean(vec_l_gen_B)})


            #get shapes of image tensors for placeholder
            # tens_img_a = tf.convert_to_tensor(images_a)
            # tens_img_b = tf.convert_to_tensor(images_b)
            # tens_img_A_fake = tf.convert_to_tensor(im_fake_A)
            # tens_img_B_fake = tf.convert_to_tensor(im_fake_B)
            #
            # self.shape_images_a = tens_img_a.get_shape()
            # self.shape_images_b = tens_img_b.get_shape()
            # self.shape_im_fake_A = tens_img_A_fake.get_shape()
            # self.shape_im_fake_B = tens_img_B_fake.get_shape()

            train_img_sum = self.sess.run(self.img_sum,
                                          feed_dict={self.img_A_sum_: images_a,
                                                     self.img_A_fake_sum_: im_fake_A,
                                                     self.img_B_sum_: images_b,
                                                     self.img_B_fake_sum_: im_fake_B})

            # Add Summaries to tensorboard
            self.summary_writer.add_summary(train_summary_metrics, epoch)
            self.summary_writer.flush()

            self.summary_writer.add_summary(val_summary_metrics, epoch)
            self.summary_writer.flush()

            self.summary_writer.add_summary(train_summary_dis, epoch)
            self.summary_writer.flush()

            self.summary_writer.add_summary(train_summary_gen, epoch)
            self.summary_writer.flush()

            self.summary_writer.add_summary(train_img_sum, epoch)
            self.summary_writer.flush()



            elapsed_time = time.time() - start_time
            print('Epoch %d took %.3f seconds' % (epoch, elapsed_time))

            # Save model
            ## TODO: save the model which performs best on validation data
            # if save:
            #     self.save(self.sess)
            #     cv2.imwrite("./Models/Images/" + self.mod_name + '/' + self.mod_name + "_Epoch_" + str(epoch) + ".png",sneak_peak[:,:,[2,1,0]]*255)
            print("")


            loss_gen_A = [np.mean(np.square(np.array(vec_ldfA))),np.mean(np.square(np.array(vec_ldfAh))),np.mean(np.array(lcA))]
            loss_gen_B = [np.mean(np.square(np.array(vec_ldfB))),np.mean(np.square(np.array(vec_ldfBh))),np.mean(np.array(lcB))]
            loss_dis_A = [np.mean(np.square(np.array(vec_ldrA))),np.mean(np.square(1.-np.array(vec_ldfA))),\
                          np.mean(np.square(np.array(vec_ldrAh))),np.mean(np.square(1.-np.array(vec_ldfAh)))]
            loss_dis_B = [np.mean(np.square(np.array(vec_ldrB))),np.mean(np.square(1.-np.array(vec_ldfB))),\
                          np.mean(np.square(np.array(vec_ldrBh))),np.mean(np.square(1.-np.array(vec_ldfBh)))]

            loss_gen_A_list.append(loss_gen_A)
            loss_gen_B_list.append(loss_gen_B)
            loss_dis_A_list.append(loss_dis_A)
            loss_dis_B_list.append((loss_dis_B))

            # establish checkpoints
            if self.checkpoints:
                if epoch % (n_epochs / 20) == 0 or epoch == n_epochs:
                    checkpoint_file = os.path.join(self.log_dir, 'model.ckpt')
                    self.saver.save(self.sess, checkpoint_file, global_step=epoch)

        final_time = time.time() - real_start_time
        print('Training took a total of {} hours'.format(final_time / 3600.0))
        # print('Number of trainable Parameters in model: ' + self.sess.run(str(count_params)))
        print('')
        print('Best Dice Score on Validation set is: {} at Epoch {}'.format(best_dice_score, best_dice_epoch))
        print('Corresponding Pearson Corr. is: {}'.format(corr_best_dice))
        print('')
        print('Best Pearson Corr. on Validation set is: {} at Epoch {}'.format(best_corr_score, best_corr_epoch))
        print('Fold: {}'.format(self.fold))

        return best_dice_score, corr_best_dice, best_corr_score  #, [loss_gen_A_list,loss_gen_B_list,loss_dis_A_list, loss_dis_B_list],

    
    def generator_A(self,batch_size=32,lambda_c=0.,lambda_h=0., checkpoint='latest', split = 'train'):

        self.log_dir = os.path.join('./logs', self.log_name, self.fold_name)

        f              = h5py.File(self.data_file,"r")
        f_save         = h5py.File("./Models/" + self.mod_name + '/' + self.mod_name + '_' + split +'_gen_A.h5',"w")
        
        # Find number of samples
        num_samples    = self.b_size
        num_iterations = num_samples // batch_size
                
        gen_data       = np.zeros((f['B/data'].shape[0],f['B/data'].shape[1],f['B/data'].shape[2],f['A/data'].shape[3]),dtype=np.uint16)

        # self.sess.run(self.init)
        with self.graph.as_default():

            self.load_weights(type=checkpoint)

            for iteration in range(num_iterations):
                images_b   = f['B/data'][(iteration*batch_size):((iteration+1)*batch_size),:,:,:]
                if images_b.dtype=='uint8':
                    images_b=images_b/float(2**8-1)
                elif images_b.dtype=='uint16':
                    images_b=images_b/float(2**16-1)
                else:
                    raise ValueError('Dataset B is not int8 or int16')

                gen_A = self.sess.run(self.fake_A,feed_dict={self.B: images_b,\
                                                        self.lambda_c: lambda_c,\
                                                        self.lambda_h: lambda_h})
                gen_data[(iteration*batch_size):((iteration+1)*batch_size),:,:,:] = (np.minimum(np.maximum(gen_A,0),1)*(2**16-1)).astype(np.uint16)

                print("\rGenerator A: {}/{} ({:.1f}%)".format(iteration+1, num_iterations, iteration*100/(num_iterations-1)),end="   ")

            group = f_save.create_group('A')
            group.create_dataset(name='data', data=gen_data,dtype=np.uint16)

            f_save.close()
            f.close()
        
        return None


    def generator_B(self,batch_size=32,lambda_c=0.,lambda_h=0.,checkpoint = 'latest', split = 'train'):

        self.log_dir = os.path.join('./logs', self.log_name, self.fold_name)

        f              = h5py.File(self.data_file,"r")
        f_save         = h5py.File("./Models/" + self.mod_name + '/' + self.mod_name + '_' + split +'_gen_B.h5',"w")
        
        # Find number of samples
        num_samples    = self.a_size
        num_iterations = num_samples // batch_size
                
        gen_data       = np.zeros((f['A/data'].shape[0],f['A/data'].shape[1],f['A/data'].shape[2],f['B/data'].shape[3]),dtype=np.uint16)
        
        # self.sess.run(self.init)
        with self.graph.as_default():

            self.load_weights(type=checkpoint)

            for iteration in range(num_iterations):
                images_a   = f['A/data'][(iteration*batch_size):((iteration+1)*batch_size),:,:,:]
                if images_a.dtype=='uint8':
                    images_a=images_a/float(2**8-1)
                elif images_a.dtype=='uint16':
                    images_a=images_a/float(2**16-1)
                else:
                    raise ValueError('Dataset A is not int8 or int16')

                gen_B = self.sess.run(self.fake_B,feed_dict={self.A: images_a,\
                                                        self.lambda_c: lambda_c,\
                                                        self.lambda_h: lambda_h})
                gen_data[(iteration*batch_size):((iteration+1)*batch_size),:,:,:] = (np.minimum(np.maximum(gen_B,0),1)*(2**16-1)).astype(np.uint16)

                print("\rGenerator B: {}/{} ({:.1f}%)".format(iteration+1, num_iterations, iteration*100/(num_iterations-1)),end="   ")

            group = f_save.create_group('B')
            group.create_dataset(name='data', data=gen_data,dtype=np.uint16)

            f_save.close()
            f.close()
        
        return None


    def predict_all(self,images_a, images_b):
        """
        Use this function to predict all image outputs, even cyclic return of same image
        :param images_a:
        :param images_b:
        :return: original images, fake images, cycled images
        """
        feed_dict = {self.A: images_a, self.B: images_b}
        im_fake_A, im_fake_B, cyc_A, cyc_B, = self.sess.run([self.fake_A, self.fake_B, self.cyc_A, self.cyc_B],
                                                            feed_dict=feed_dict)

        return images_a, images_b, im_fake_A, im_fake_B, cyc_A, cyc_B


    def predict_seg(self, images_a, images_b, heart_data=False):
        """
        Only predict segmentation results
        :param images_a:
        :param images_b:
        :return: f1_score for batch (= Dice score)
        """
        with self.graph.as_default():
            # get fake image batch
            images_b = images_b
            im_fake_B = self.sess.run(self.fake_B, feed_dict={self.A: images_a})

            im_fake_B = (np.minimum(np.maximum(im_fake_B,0),1)*(2**8-1)).astype(np.uint8)
            # print('Non Rounded B')
            # print(images_b[0,50:65,50:65,:])
            # print('Non Rounded Fake B')
            # print(im_fake_B[0,50:65,50:65,:])

            # self.lambda_c: lambda_c,
            # self.lambda_h: lambda_h}
            # perform rounding since generated values are between 0 - 255 (roughly 0, 120, 255), might have some slight deviations there
            # --> map it to values: 0 (Coll) / 1 (Cells) / 2 (Void)
            # can't properly evaluate F1 score on raw images, left out here

            if heart_data:
                nr_labels = 3
                rd_b = np.rint(images_b / 120)
                rd_fk_b = np.rint(im_fake_B / 120)
            else:
                raise ValueError("Currently only heart_data is supported for validation")

            # print('Rounded B')
            # print(rd_b[0,50:65,50:65,:])
            # print('Rounded Fake B')
            # print(rd_fk_b[0,50:65,50:65,:])
            # assert rd_a.shape[0] == rd_b.shape[0] == rd_fk_a.shape[0] == rd_fk_b.shape[0] == 2
            # assert rd_a.shape[1] == rd_b.shape[1] == rd_fk_a.shape[1] == rd_fk_b.shape[1]
            # assert rd_a.shape[2] == rd_b.shape[2] == rd_fk_a.shape[2] == rd_fk_b.shape[2]

            # dim1 = rd_a.shape[1]
            # dim2 = rd_a.shape[2]

            # If Sklearn's f1 score is used, have to reshape arrays with: .reshape(dim1*dim2)
            # left out A (raw) images since they are not relevant for prediction of F1 score
            # for i in range(rd_b.shape[0]):
            #     # im_a += list(rd_a[i,:,:,:])
            #     im_b += list(rd_b[i, :, :, :])
            #     # fake_im_a += list(rd_fk_a[i,:,:,:])
            #     fake_im_b += list(rd_fk_b[i, :, :, :])

            # f1_macro = f1_score(np.asarray(im_b),np.asarray(fake_im_b), average='macro')



            f1_macro = self.sess.run([self.get_mean_dice], feed_dict={self.fake_B: rd_fk_b,
                                                                      self.B: rd_b,
                                                                      self.n_labels: nr_labels})

            flat_b = rd_b.flatten()
            flat_fk_b = rd_fk_b.flatten()
            sk_f1_macro = f1_score(flat_b, flat_fk_b,average='macro')

            # obtain pearson corr for collagen fraction
            if heart_data:
                frac_list_b = []
                frac_list_fake_b = []
                epsi = 1e-10
                for i in range(rd_b.shape[0]):
                    coll_b = np.sum(rd_b[i, :, :, :] == 0)
                    coll_fake_b = np.sum(rd_fk_b[i, :, :, :] == 0)

                    cells_b = np.sum(rd_b[i, :, :, :] == 1)
                    cells_fake_b = np.sum(rd_fk_b[i, :, :, :] == 1)

                    fraction_b = coll_b / (epsi + cells_b)
                    fraction_fake_b = coll_fake_b / (epsi + cells_fake_b)

                    frac_list_b.append(fraction_b)
                    frac_list_fake_b.append(fraction_fake_b)


        return f1_macro, frac_list_b, frac_list_fake_b, sk_f1_macro


    def do_validation(self, iterator):
        """
        Provides various metrics of interest
        :param iterator: iterator over batches
        :return: average batch losses and other relevant metrics, e.g. F1 score
        """
        with self.graph.as_default():
            dice_score_aggr = 0
            sk_dice = 0
            frac_list_b = []
            frac_list_fake_b = []
            num_batches = 0

            for batch in iterator(self.batch_size):
                batch_a, batch_b = batch

                assert batch_a.shape[0] == batch_b.shape[0]

                if batch_a.shape[0] < self.batch_size:
                    continue

                # images_a, images_b, im_fake_A, im_fake_B, cyc_A, cyc_B = self.predict(batch_a, batch_b)
                f1_macro, coll_frac_b, coll_frac_fake_b, sk_f1_macro = self.predict_seg(batch_a, batch_b, heart_data=True)

                # make sure only 1 element is in list so we can add it
                assert len(f1_macro) == 1

                #dice score and collagen fractions
                dice_score_aggr += f1_macro[0]
                sk_dice += sk_f1_macro
                frac_list_b += coll_frac_b
                frac_list_fake_b += coll_frac_fake_b

                #get pearson correlation for the collagen fractions
                corr, _ = pearsonr(frac_list_b, frac_list_fake_b)

                num_batches += 1

            # get the mean dice score for the dataset
            mean_dice_score = dice_score_aggr / num_batches
            mean_sk_dice = sk_dice / num_batches
            # print('### B Frac ###')
            # print(frac_list_b)
            # print('### Fake B Frac ###')
            # print(frac_list_fake_b)
        return mean_dice_score, frac_list_b, frac_list_fake_b, corr, mean_sk_dice


    def test(self, batch_size=1, num_sample_volumes=2, checkpoint = 'best_dice', heart_data=True, datatype='test',
             gen_img=False, group_b = None, group_fake_b=None):
        """
        Used when testing the trained model, calculates the defined metrics for the input set (train / val / test) and has option
        to create new fake datasets.
        :param batch_size:
        :param num_sample_volumes:
        :param checkpoint:
        :param heart_data:
        :param datatype:
        :param gen_img:
        :return:
        """
        self.log_dir = os.path.join('./logs', self.log_name, self.fold_name)
        heart_data = heart_data
        dtype = np.uint8

        summary_dict = {}
        summary_dict['nr_samples'] = num_sample_volumes

        if datatype == 'train':
            print('INFO:   Evaluating Test results for Training data')
            print('')
        elif datatype == 'validation':
            print('INFO:   Evaluating Test results for Validation data')
            print('')
        elif datatype == 'test':
            print('INFO:   Evaluating Test results for Test data')
            print('')
        else:
            raise ValueError('Need to specify either train, validation or test split')

        if gen_img:
            fold_grp = group_b.create_group(self.fold_name)
            fake_fold_grp = group_fake_b.create_group(self.fold_name)

        with self.graph.as_default():
            self.load_weights(type=checkpoint)

            num_samples = self.data.nr_images * self.data.aug_factor
            num_iterations = num_samples // batch_size

            tot_dice = 0
            tot_corr = 0
            sample_vol_idx_list = []

            for sample in range(num_sample_volumes):
                print('INFO:   Currently iterating through sample volume {} (non-coded)'.format(sample))
                dice = 0
                frac_list_b = []
                frac_list_fake_b = []
                num_batches = 0

                data_B = np.zeros(shape=[num_samples, self.imsize, self.imsize, self.b_chan], dtype=np.uint8)
                data_fake_B = np.zeros(shape=[num_samples, self.imsize, self.imsize, self.b_chan], dtype=np.uint8)

                # generate nested dict with information
                summary_dict[sample] = {}

                for iteration in range(int(num_iterations)):

                    if iteration % 100 == 0:
                        print('INFO:   Currently at iteration {} / {}'.format(iteration, int(num_iterations)))

                    # iterate also over number of sample volumes
                    if datatype == 'train':
                        images_a, images_b, sample_vol_idx = self.train_data.test_image(img_idx=iteration,
                                                                                       batch_size=batch_size,
                                                                                       sample_vol=sample)
                    elif datatype == 'validation':
                        images_a, images_b, sample_vol_idx = self.validation_data.test_image(img_idx=iteration,
                                                                                       batch_size=batch_size,
                                                                                       sample_vol=sample)
                    elif datatype == 'test':
                        images_a, images_b, sample_vol_idx = self.test_data.test_image(img_idx=iteration,
                                                                                       batch_size=batch_size,
                                                                                       sample_vol=sample)

                    assert images_a.shape[0] == images_b.shape[0] == 1, print('Shape should be 1 in first dimension')

                    # if images_a.dtype=='uint8':
                    #     images_a=images_a/float(2**8-1)
                    # elif images_a.dtype=='uint16':
                    #     images_a=images_a/float(2**16-1)
                    # else:
                    #     raise ValueError('Dataset A is not int8 or int16')
                    # if images_b.dtype=='uint8':
                    #     images_b=images_b/float(2**8-1)
                    # elif images_b.dtype=='uint16':
                    #     images_b=images_b/float(2**16-1)
                    # else:
                    #     raise ValueError('Dataset B is not int8 or int16')

                    im_fake_B = self.sess.run(self.fake_B, feed_dict={self.A: images_a})
                    im_fake_B = (np.minimum(np.maximum(im_fake_B, 0), 1) * (2 ** 8 - 1)).astype(np.uint8)

                    if heart_data:
                        rd_b = np.rint(images_b / 120)
                        rd_fk_b = np.rint(im_fake_B / 120)
                    else:
                        raise ValueError("Currently only heart_data is supported for validation")


                    if gen_img:
                        data_fake_B[iteration, :, :, :] = rd_fk_b
                        data_B[iteration,:,:,:] = rd_b

                    flat_b = rd_b.flatten()
                    flat_fk_b = rd_fk_b.flatten()

                    sk_f1_macro = f1_score(flat_b, flat_fk_b, average='macro')

                    # obtain pearson corr for collagen fraction
                    if heart_data:
                        epsi = 1e-10
                        assert rd_b.shape[0] == 1

                        for i in range(rd_b.shape[0]):
                            coll_b = np.sum(rd_b[i, :, :, :] == 0)
                            coll_fake_b = np.sum(rd_fk_b[i, :, :, :] == 0)

                            cells_b = np.sum(rd_b[i, :, :, :] == 1)
                            cells_fake_b = np.sum(rd_fk_b[i, :, :, :] == 1)

                            fraction_b = coll_b / (epsi + cells_b)
                            fraction_fake_b = coll_fake_b / (epsi + cells_fake_b)


                    dice += sk_f1_macro
                    frac_list_b.append(fraction_b)
                    frac_list_fake_b.append(fraction_fake_b)

                    num_batches += 1

                    # get the mean dice score for the dataset
                mean_dice_score = dice / num_batches
                corr, _ = pearsonr(frac_list_b, frac_list_fake_b)

                print('INFO:  Mean Dice Score of Sample {} is: {}'.format(sample, mean_dice_score))
                print('INFO:  Mean Pearson Corr. of Sample {} is: {}'.format(sample, corr))

                sample_vol_idx_list.append(sample_vol_idx)

                summary_dict[sample]['dice'] = mean_dice_score
                summary_dict[sample]['corr'] = corr
                summary_dict[sample]['real_frac'] = frac_list_b
                summary_dict[sample]['fake_frac'] = frac_list_fake_b
                summary_dict[sample]['sample_idx'] = sample_vol_idx
                tot_dice += mean_dice_score
                tot_corr += corr

                if gen_img:
                    fold_grp.create_dataset(name='data_' + str(sample_vol_idx), data=data_B, dtype=dtype)
                    fake_fold_grp.create_dataset(name='data_' + str(sample_vol_idx), data=data_fake_B, dtype=dtype)

            tot_dice = tot_dice / num_sample_volumes
            tot_corr = tot_corr / num_sample_volumes

            print('INFO:  Total Mean Dice: {}'.format(tot_dice))
            print('INFO:  Total Mean Pearson Corr: {}'.format(tot_corr))

        return summary_dict


    #loading weights and other stuff, function from Christian Baumgartner's discriminative learning toolbox
    def load_weights(self, log_dir=None, type='latest', **kwargs):

        if not log_dir:
            log_dir = self.log_dir

        if type == 'latest':
            init_checkpoint_path = tf_utils.get_latest_model_checkpoint_path(log_dir, 'model.ckpt')
        elif type == 'best_dice':
            init_checkpoint_path = tf_utils.get_latest_model_checkpoint_path(log_dir, 'model_best_dice.ckpt')
        else:
            raise ValueError('Argument type=%s is unknown. type can be latest/best_dice' % type)

        print('Loaded checkpoint of type ' + str(type) + ' from log directory ' + str(self.log_dir))
        self.saver.restore(self.sess, init_checkpoint_path)


    # Helper Functions
    def _make_tensorboard_summaries(self):
        with self.graph.as_default():
            # self.rel_lr_ = tf.placeholder(tf.float32, shape= [], name= 'rel_lr_')
            # self.rel_lr_sum = tf.summary.scalar('learning_rate', self.rel_lr_)
            # Build the summary Tensor based on the TF collection of Summaries.

            ## Dice Score
            self.train_mean_dice_summary_ = tf.placeholder(tf.float32, shape=[], name='train_mean_dice')
            train_mean_dice_summary = tf.summary.scalar('Mean_Dice_Training', self.train_mean_dice_summary_)
            self.val_mean_dice_summary_ = tf.placeholder(tf.float32, shape=[], name='val_mean_dice')
            val_mean_dice_summary = tf.summary.scalar('Mean_Dice_Validation', self.val_mean_dice_summary_)

            ## Pearson Correlation
            self.train_corr_summary_ = tf.placeholder(tf.float32, shape=[], name='train_corr')
            train_corr_summary = tf.summary.scalar('Pearson_Corr_Training', self.train_corr_summary_)
            self.val_corr_summary_ = tf.placeholder(tf.float32, shape=[], name='val_corr')
            val_corr_summary = tf.summary.scalar('Pearson_Corr_Validation', self.val_corr_summary_)

            ## Dice score calculated with Sklearn's library
            self.train_sk_dice_summary_ = tf.placeholder(tf.float32, shape=[], name='train_sk_dice')
            train_sk_dice_summary = tf.summary.scalar('sk_dice_Training', self.train_sk_dice_summary_)
            self.val_sk_dice_summary_ = tf.placeholder(tf.float32, shape=[], name='val_sk_dice')
            val_sk_dice_summary = tf.summary.scalar('sk_dice_Validation', self.val_sk_dice_summary_)

            ## Collagen Fractions
            self.val_coll_frac_real_summary_ = tf.placeholder(tf.float32, shape=[self.data.nr_images], name='val_coll_frac_real')
            val_coll_frac_real_summary = tf.summary.histogram('Val_Frac_Real', self.val_coll_frac_real_summary_)
            self.val_coll_frac_fake_summary_ = tf.placeholder(tf.float32, shape=[self.data.nr_images], name='val_coll_frac_fake')
            val_coll_frac_fake_summary = tf.summary.histogram('Val_Frac_Fake', self.val_coll_frac_fake_summary_)

            self.train_coll_frac_real_summary_ = tf.placeholder(tf.float32, shape=[self.data.nr_images], name='train_coll_frac_real')
            train_coll_frac_real_summary = tf.summary.histogram('train_Frac_Real', self.train_coll_frac_real_summary_)
            self.train_coll_frac_fake_summary_ = tf.placeholder(tf.float32, shape=[self.data.nr_images], name='train_coll_frac_fake')
            train_coll_frac_fake_summary = tf.summary.histogram('train_Frac_Fake', self.train_coll_frac_fake_summary_)

            # Merging scalar summaries
            self.train_summary = tf.summary.merge([train_mean_dice_summary, train_corr_summary, train_sk_dice_summary,
                                                   train_coll_frac_real_summary, train_coll_frac_fake_summary])
            self.val_summary = tf.summary.merge([val_mean_dice_summary, val_corr_summary, val_sk_dice_summary,
                                                 val_coll_frac_real_summary, val_coll_frac_fake_summary])

            # Losses
            self.train_loss_dis_A_summary_ = tf.placeholder(tf.float32, shape=[], name='l_D_A')
            loss_dis_A_summary = tf.summary.scalar('Loss_of_Discr_A', self.train_loss_dis_A_summary_)

            self.train_loss_dis_B_summary_ = tf.placeholder(tf.float32, shape=[], name='l_D_B')
            loss_dis_B_summary = tf.summary.scalar('Loss_of_Discr_B', self.train_loss_dis_B_summary_)

            self.train_loss_gen_A_summary_ = tf.placeholder(tf.float32, shape=[], name='l_G_A')
            loss_gen_A_summary = tf.summary.scalar('Loss_of_Gen_A', self.train_loss_gen_A_summary_)

            self.train_loss_gen_B_summary_ = tf.placeholder(tf.float32, shape=[], name='l_G_B')
            loss_gen_B_summary = tf.summary.scalar('Loss_of_Gen_B', self.train_loss_gen_B_summary_)

            self.train_loss_dis = tf.summary.merge([loss_dis_A_summary, loss_dis_B_summary])
            self.train_loss_gen = tf.summary.merge([loss_gen_A_summary, loss_gen_B_summary])

            # Images
            self.img_A_sum_ = tf.placeholder(tf.float32, shape= [None, None, None, None], name = 'img_A_pl') #self.batch_size, 256, 256, 1
            img_A_sum = tf.summary.image('Image_A', self.img_A_sum_, max_outputs=1)

            self.img_A_fake_sum_ = tf.placeholder(tf.float32, shape= [None, None, None, None], name = 'img_A_fake_pl')
            img_A_fake_sum = tf.summary.image('Image_A_fake', self.img_A_fake_sum_, max_outputs=1)

            self.img_B_sum_ = tf.placeholder(tf.float32, shape= [None, None, None, None], name = 'img_B_pl')
            img_B_sum = tf.summary.image('Image_B', self.img_B_sum_, max_outputs=1)

            self.img_B_fake_sum_ = tf.placeholder(tf.float32, shape= [None, None, None, None], name = 'img_B_fake_pl')
            img_B_fake_sum = tf.summary.image('Image_B_fake', self.img_B_fake_sum_, max_outputs=1)

            self.img_sum = tf.summary.merge([img_A_sum, img_A_fake_sum, img_B_sum, img_B_fake_sum])


    def _setup_log_dir_and_continue_mode(self):

        # Default values
        self.log_dir = os.path.join('./logs', self.log_name, self.fold_name)
        self.init_checkpoint_path = None
        self.continue_run = False
        self.init_step = 0

        if self.checkpoints:
            # If a checkpoint file already exists enable continue mode
            if tf.gfile.Exists(self.log_dir):
                init_checkpoint_path = tf_utils.get_latest_model_checkpoint_path(self.log_dir, 'model.ckpt')
                if init_checkpoint_path is not False:
                    self.init_checkpoint_path = init_checkpoint_path
                    self.continue_run = True
                    self.init_step = int(self.init_checkpoint_path.split('/')[-1].split('-')[-1])
                    self.log_dir += '_cont'

                    logging.info(
                        '--------------------------- Continuing previous run --------------------------------')
                    logging.info('Checkpoint path: %s' % self.init_checkpoint_path)
                    logging.info('Latest step was: %d' % self.init_step)
                    logging.info(
                        '------------------------------------------------------------------------------------')

        tf.gfile.MakeDirs(self.log_dir)
        self.summary_writer = tf.summary.FileWriter(self.log_dir, self.graph)
        #
        # # Copy experiment config file to log_dir for future reference
        # shutil.copy(self.exp_config.__file__, self.log_dir)

    def show_params(self):
        with self.graph.as_default():
            total = 0
            for v in tf.trainable_variables():
                dims = v.get_shape().as_list()
                num = int(np.prod(dims))
                total += num
                print('  %s \t\t Num: %d \t\t Shape %s ' % (v.name, num, dims))
            print('\nTotal number of params: %d' % total)


    def corr_plots(self, train_real_fraction, train_fake_fraction, val_real_fraction, val_fake_fraction):
        fig, axes = plt.subplots(1, 2)

        train_real_fraction = np.asarray(train_real_fraction)
        train_fake_fraction = np.asarray(train_fake_fraction)
        val_real_fraction = np.asarray(val_real_fraction)
        val_fake_fraction = np.asarray(val_fake_fraction)

        axes[0].plot(np.arange(100), train_real_fraction[0:100], '-b')
        axes[0].plot(np.arange(100), train_fake_fraction[0:100], '-r')
        axes[0].legend('Real Images', 'Fake Images')
        axes[0].set_title('Training')

        axes[1].plot(np.arange(100), val_real_fraction[0:100],'-b')
        axes[1].plot(np.arange(100), val_fake_fraction[0:100], '-r')
        axes[1].legend('Real Images', 'Fake Images')
        axes[1].set_title('Validation')

        plt.ion()
        plt.show()
        plt.pause(0.001)

    ######## Currently unused functions, left over from initial implementation ########

    # def save(self,sess):
    #     """
    #     Save the model parameter in a ckpt file. The filename is as
    #     follows:
    #     ./Models/<mod_name>.ckpt
    #
    #     INPUT: sess         - The current running session
    #     """
    #     self.saver.save(sess,"./Models/" + self.mod_name + '/' + self.mod_name + ".ckpt")
    #
    # def init(self,sess):
    #     """
    #     Init the model. If the model exists in a file, load the model. Otherwise, initalize the variables
    #
    #     INPUT: sess         - The current running session
    #     """
    #     if not os.path.isfile(\
    #             "./Models/" + self.mod_name + '/' + self.mod_name + ".ckpt.meta"):
    #         sess.run(tf.global_variables_initializer())
    #         return 0
    #     else:
    #         if self.gen_only:
    #             sess.run(tf.global_variables_initializer())
    #         self.load(sess)
    #         return 1
    #
    # def load(self,sess):
    #     """
    #     Load the model from the parameter file:
    #     ./Models/<mod_name>.ckpt
    #
    #     INPUT: sess         - The current running session
    #     """
    #     self.saver.restore(sess, "./Models/" + self.mod_name + '/' + self.mod_name + ".ckpt")

    # def get_loss(self, lambda_c=0., lambda_h=0.):
    #     f = h5py.File(self.data_file, "r")
    #
    #     rand_a = np.random.randint(self.a_size - 32)
    #     rand_b = np.random.randint(self.b_size - 32)
    #
    #     images_a = f['A/data'][rand_a:(rand_a + 32), :, :, :] / 255.
    #     images_b = f['B/data'][rand_b:(rand_b + 32), :, :, :] / 255.
    #
    #     self.sess.run(tf.global_variables_initializer())
    #
    #     l_rA, l_rB, l_fA, l_fB = \
    #         self.sess.run([self.dis_real_A, self.dis_real_B, self.dis_fake_A, self.dis_fake_B, ], \
    #                       feed_dict={self.A: images_a, \
    #                                  self.B: images_b, \
    #                                  self.lambda_c: lambda_c, \
    #                                  self.lambda_h: lambda_h})
    #
    #     f.close()
    #     return l_rA, l_rB, l_fA, l_fB
    #
    # def predict(self, lambda_c=0., lambda_h=0.):
    #     f = h5py.File(self.data_file, "r")
    #
    #     rand_a = np.random.randint(self.a_size - 32)
    #     rand_b = np.random.randint(self.b_size - 32)
    #
    #     images_a = f['A/data'][rand_a:(rand_a + 32), :, :, :] / 255.
    #     images_b = f['B/data'][rand_b:(rand_b + 32), :, :, :] / 255.
    #
    #     self.sess.run(tf.global_variables_initializer())
    #
    #     fake_A, fake_B, cyc_A, cyc_B = \
    #         self.sess.run([self.fake_A, self.fake_B, self.cyc_A, self.cyc_B], \
    #                       feed_dict={self.A: images_a, \
    #                                  self.B: images_b, \
    #                                  self.lambda_c: lambda_c, \
    #                                  self.lambda_h: lambda_h})
    #
    #     f.close()
    #     return images_a, images_b, fake_A, fake_B, cyc_A, cyc_B

