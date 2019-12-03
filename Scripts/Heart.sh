#!/bin/sh

# Create a Directory for model data ; make sure you have directories in Models, logs and Images
mkdir -p Models

# Train the network
python main.py --dataset=./data_processing/aug_heart_data.h5 --name=Heart_aug --log_name=Heart_aug --fold=1

#generate stuff, specify the log_name for log directory (files for checkpoints etc) and which type of checkpoint you want (e.g. latest or best_f1)
python main.py --dataset=./Data/Heart/heart_train_set_noisy.h5 --name=Heart_noisy --log_name=Heart_noisy --mode=gen_B --checkpoint=latest

#generate stuff for test set
python main.py --dataset=./Data/Heart/heart_test_set_noisy.h5 --name=Heart_noisy --log_name=Heart_noisy --mode=gen_B --checkpoint=latest --split=test

#adapt parameters (filepaths) here first before running it --> creates png images from h5 file
python h5_to_img.py