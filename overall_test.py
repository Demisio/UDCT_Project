import cycleGAN
import re
import sys
from os import environ as cuda_environment
import os
import numpy as np
import tensorflow as tf
import time
import pickle
import gc
import h5py

## Modify batch_size

if __name__ == "__main__":

    save_path = './Results/Heart_limited'

    #Modify some parameters
    heart_data = True
    gen_img = False
    if gen_img:
        print('')
        print('INFO:   Variable gen_img is set to True, Images and metrics will be analysed')
        print('')
    else:
        print('')
        print('INFO:   Variable gen_img is set to False, no Images will be generated. Only metrics will be calculated')
        print('')

    # List of floats
    sub_value_f = {}
    sub_value_f['lambda_c'] = 10.  # Loss multiplier for cycle
    sub_value_f['lambda_h'] = 1.  # Loss multiplier for histogram
    sub_value_f['dis_noise'] = 0.1  # Std of gauss noise added to Dis
    sub_value_f['syn_noise'] = 0.  # Add gaussian noise to syn images to make non-flat backgrounds
    sub_value_f['real_noise'] = 0.  # Add gaussian noise to real images to make non-flat backgrounds

    # List of ints
    sub_value_i = {}
    sub_value_i['epoch'] = 200  # Number of epochs to be trained
    sub_value_i['batch_size'] = 2  # Batch size for training
    sub_value_i['buffer_size'] = 50  # Number of history elements used for Dis
    sub_value_i['save'] = 1  # If not 0, model is saved
    sub_value_i['gpu'] = 0  # Choose the GPU ID (if only CPU training, choose nonexistent number)
    sub_value_i['verbose'] = 0  # If not 0, some network information is being plotted
    sub_value_i['fold'] = 1  # Which fold is in use?
    sub_value_i['checkpoints'] = 0  # Do checkpoints?

    # List of strings
    sub_string = {}
    sub_string['name'] = 'unnamed'  # Name of model (should be unique). Is used to save/load models
    sub_string['dataset'] = 'pathtodata.h5'  # Describes which h5 file is used
    sub_string['architecture'] = 'Res6'  # Network architecture: 'Res6' or 'Res9'
    sub_string['deconv'] = 'transpose'  # Upsampling method: 'transpose' or 'resize'
    sub_string['PatchGAN'] = 'Patch70'  # Choose the Gan type: 'Patch34', 'Patch70', 'Patch142', 'MultiPatch'
    sub_string['mode'] = 'training'  # 'train', 'gen_A', 'gen_B'
    sub_string['log_name'] = 'logs'  # log file directory
    sub_string['checkpoint'] = 'best_dice'  # which checkpoint should be loaded for generators at test time 'latest' / 'best_f1'
    sub_string['split'] = 'train'  # which split do you use (labelling of created dataset only)

    # Create complete dictonary
    var_dict = sub_string.copy()
    var_dict.update(sub_value_i)
    var_dict.update(sub_value_f)

    # Update all defined parameters in dictionary
    for arg_i in sys.argv[1:]:
        var = re.search('(.*)\=', arg_i)  # everything before the '='
        g_var = var.group(1)[2:]
        if g_var in sub_value_i:
            dtype = 'int'
        elif g_var in sub_value_f:
            dtype = 'float'
        elif g_var in sub_string:
            dtype = 'string'
        else:
            print("Unknown key word: " + g_var)
            print("Write parameters as: <key word>=<value>")
            print("Example: 'python main.py buffer_size=32'")
            print("Possible key words: " + str(var_dict.keys()))
            continue

        content = re.search('\=(.*)', arg_i)  # everything after the '='
        g_content = content.group(1)
        if dtype == 'int':
            var_dict[g_var] = int(g_content)
        elif dtype == 'float':
            var_dict[g_var] = float(g_content)
        else:
            var_dict[g_var] = g_content
    if not os.path.isfile(var_dict['dataset']):
        raise ValueError('Dataset does not exist. Specify loation of an existing h5 file.')
    # Get the dataset filename

    # Restrict usage of GPUs
    cuda_environment["CUDA_VISIBLE_DEVICES"] = str(var_dict['gpu'])
    with open('Models/' + var_dict['name'] + '/' + 'fold_' + str(var_dict['fold']) + '/' + var_dict[
        'name'] + "_params.txt", "w") as myfile:
        for key in sorted(var_dict):
            myfile.write(key + "," + str(var_dict[key]) + "\n")

    # Find out, if whole network is needed or only the generators
    gen_only = False
    if 'gen' in var_dict['mode']:
        gen_only = True

    # for later use
    group_b = None
    group_fake_b = None
    num_sample_volumes = None

    real_start_time = time.time()
    # Define the model
    for fold in range (4,5):
        start_time = time.time()

        #same model as in main but use fold=fold instead of an input
        model = cycleGAN.Model(mod_name=var_dict['name'],
                               data_file=var_dict['dataset'],
                               buffer_size=var_dict['buffer_size'],
                               dis_noise=var_dict['dis_noise'],
                               architecture=var_dict['architecture'],
                               lambda_c=var_dict['lambda_c'],
                               lambda_h=var_dict['lambda_h'],
                               deconv=var_dict['deconv'],
                               patchgan=var_dict['PatchGAN'],
                               verbose=(var_dict['verbose'] != 0),
                               gen_only=gen_only,
                               log_name=var_dict['log_name'],
                               fold=fold,
                               checkpoints=var_dict['checkpoints'])

        assert (var_dict['mode'] == 'test') or (var_dict['mode'] == 'validation') or (var_dict['mode'] == 'train'), \
            ('Info: Please use \'--mode=test\', \'--mode=train\' or \'--mode=validation\' when calling the program')

        print('Chosen set for fold {} is : {}'.format(fold, var_dict['mode']))
        print('')
        print('INFO:   Beginning testing phase')


        # specific for heart data currently
        if heart_data:
            if var_dict['mode'] == 'test':
                num_sample_volumes = 2
                if gen_img and fold == 1:
                    filename = './Results/' + var_dict['log_name'] +'/Images/pred_test.h5'
                    f = h5py.File(filename, "w")
                    group_b = f.create_group('B_real')
                    group_fake_b = f.create_group('B_fake')

            elif var_dict['mode'] == 'validation':
                num_sample_volumes = 2
                if gen_img and fold == 1:
                    filename = './Results/' + var_dict['log_name'] +'/Images/pred_validation.h5'
                    f = h5py.File(filename, "w")
                    group_b = f.create_group('B_real')
                    group_fake_b = f.create_group('B_fake')

            elif var_dict['mode'] == 'train':
                num_sample_volumes = 6
                if gen_img and fold == 1:
                    filename = './Results/' + var_dict['log_name'] +'/Images/pred_train.h5'
                    f = h5py.File(filename, "w")
                    group_b = f.create_group('B_real')
                    group_fake_b = f.create_group('B_fake')
            else:
                raise ValueError('Need to specify either train, validation or test split')

        summary_dict = model.test(
                                  batch_size=1,
                                  num_sample_volumes=num_sample_volumes,
                                  checkpoint=var_dict['checkpoint'],
                                  heart_data=heart_data,
                                  gen_img=gen_img,
                                  datatype=var_dict['mode'],
                                  group_b=group_b,
                                  group_fake_b=group_fake_b)

        if var_dict['mode'] == 'test':
            pickle.dump(summary_dict, open(os.path.join(save_path, 'test_summary_dicts_fold{}.p'.format(fold)), 'wb'))
        elif var_dict['mode'] == 'validation':
            pickle.dump(summary_dict, open(os.path.join(save_path, 'validation_summary_dicts_fold{}.p'.format(fold)), 'wb'))
        elif var_dict['mode'] == 'train':
            pickle.dump(summary_dict, open(os.path.join(save_path, 'train_summary_dicts_fold{}.p'.format(fold)), 'wb'))


        elapsed_time = time.time() - start_time
        print('Predicting fold %d took %.3f seconds' % (fold, elapsed_time))

        del model
        gc.collect()

    total_elapsed_time = time.time() - real_start_time
    print('Total elapsed time: %.3f minutes' % (elapsed_time / 60))


# with open(os.path.join('Models', var_dict['name'], 'fold_' + str(var_dict['fold']), 'best_dice'), 'w') as file:
#     file.write('Best dice Score in Fold {}: {}' + '\n'.format(var_dict['fold'], best_dice))
#     file.write('Corresponding Pearson Corr: {}' + '\n'.format(corr_best_dice))
#     file.write('Best Pearson Corr: {}' + '\n'.format(best_corr_score))


# if var_dict['mode'] == 'gen_A':
#     model.generator_A(batch_size=var_dict['batch_size'],
#                       lambda_c=var_dict['lambda_c'],
#                       lambda_h=var_dict['lambda_h'],
#                       split=var_dict['split'])
#
# elif var_dict['mode'] == 'gen_B':
#     model.generator_B(batch_size=var_dict['batch_size'],
#                       lambda_c=var_dict['lambda_c'],
#                       lambda_h=var_dict['lambda_h'],
#                       checkpoint=var_dict['checkpoint'],
#                       split=var_dict['split'])
