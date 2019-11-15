#!/bin/sh

# Create the sample dataset
cd ..
python create_h5_dataset.py ./Data/Heart/06_WK1_03_Cropabs/ ./Data/Heart/06_WK1_03_Fusion/ ./Data/Heart/heart_train_set.h5

# Create a Directory for model data ; make sure you have directories in Models, logs and Images
mkdir -p Models

# Train the network
python main.py --dataset=./Data/Heart/heart_train_set.h5 --name=Heart --log_name=Heart


#generate stuff, specify the log_name for log directory (files for checkpoints etc) and which type of checkpoint you want (e.g. latest or best_f1)
python main.py --dataset=./Data/Example/example_dataset.h5 --name=test_example_2 --mode=gen_B --log_name=test_example_2 --checkpoint=latest

#adapt parameters (filepaths) here first before running it --> creates png images from h5 file
python h5_to_img.py