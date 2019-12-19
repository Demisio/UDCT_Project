#!/bin/sh

# Create a Directory for model data ; make sure you have directories in Models, logs and Images
mkdir -p Models

# Train the network
python main.py --dataset=./data_processing/aug_heart_data.h5 --name=Heart_val --log_name=Heart_val --fold=1
python main.py --dataset=./data_processing/aug_heart_data_noisy.h5 --name=Heart_val_noisy --log_name=Heart_val_noisy --fold=1
python main.py --dataset=./data_processing/aug_heart_data_noisy.h5 --name=Heart_full --log_name=Heart_full --fold=1

#train with limited data
python main.py --dataset=./data_processing/aug_heart_lim_data.h5 --name=Heart_limited --log_name=Heart_limited --fold=4


#generate stuff, specify the log_name for log directory (files for checkpoints etc) and which type of checkpoint you want (e.g. latest or best_f1)
python main.py --dataset=./Data/Heart/heart_train_set_noisy.h5 --name=Heart_noisy --log_name=Heart_noisy --mode=gen_B --checkpoint=latest

#generate stuff for test set
python overall_test.py --dataset=./data_processing/aug_heart_data_test.h5 --name=Heart_full --log_name=Heart_full --mode=test
python overall_test.py --dataset=./data_processing/aug_heart_data_test.h5 --name=Heart_full --log_name=Heart_full --mode=validation
python overall_test.py --dataset=./data_processing/aug_heart_data_test.h5 --name=Heart_full --log_name=Heart_full --mode=train

python overall_test.py --dataset=./data_processing/aug_heart_data_test.h5 --name=Heart_limited --log_name=Heart_limited --mode=test --checkpoint=latest
python overall_test.py --dataset=./data_processing/aug_heart_data_test.h5 --name=Heart_limited --log_name=Heart_limited --mode=validation --checkpoint=latest
python overall_test.py --dataset=./data_processing/aug_heart_data_test.h5 --name=Heart_limited --log_name=Heart_limited --mode=train --checkpoint=latest




python main.py --dataset=./Data/Heart/heart_test_set_noisy.h5 --name=Heart_noisy --log_name=Heart_noisy --mode=gen_B --checkpoint=latest --split=test

#adapt parameters (filepaths) here first before running it --> creates png images from h5 file
python h5_to_img.py