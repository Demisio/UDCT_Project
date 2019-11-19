#!/bin/sh

#create png images, possibly add noise and potentially crop
# params: Image path / Path for new images / Add noise? (noise / no_noise) / Crop (crop / no_crop)
python add_noise_create_png.py ./Data/Heart/06_WK1_03_Fusion/ ./Data/Heart/06_WK1_03_Fusion_crop/ no_noise crop

# Create the sample dataset
cd ..
python create_h5_dataset.py ./Data/Heart/06_WK1_03_Cropabs/ ./Data/Heart/06_WK1_03_Fusion/ ./Data/Heart/heart_train_set.h5
python create_h5_dataset.py ./Data/Heart/06_WK1_03_Cropabs_png/ ./Data/Heart/06_WK1_03_Fusion_noisy/ ./Data/Heart/heart_train_set_noisy.h5

#create test h5
python create_h5_dataset.py ./Data/Heart/06_WK1_03_Cropabs_test_png/ ./Data/Heart/06_WK1_03_Fusion_test_noisy/ ./Data/Heart/heart_test_set_noisy.h5

# Create a Directory for model data ; make sure you have directories in Models, logs and Images
mkdir -p Models

# Train the network
python main.py --dataset=./Data/Heart/heart_train_set.h5 --name=Heart --log_name=Heart
python main.py --dataset=./Data/Heart/heart_train_set_noisy.h5 --name=Heart_noisy --log_name=Heart_noisy

#generate stuff, specify the log_name for log directory (files for checkpoints etc) and which type of checkpoint you want (e.g. latest or best_f1)
python main.py --dataset=./Data/Heart/heart_train_set.h5 --name=Heart --log_name=Heart --mode=gen_B --checkpoint=latest
python main.py --dataset=./Data/Heart/heart_train_set_noisy.h5 --name=Heart_noisy --log_name=Heart_noisy --mode=gen_B --checkpoint=latest

#generate stuff for test set
python main.py --dataset=./Data/Heart/heart_test_set_noisy.h5 --name=Heart_noisy --log_name=Heart_noisy --mode=gen_B --checkpoint=latest --split=test

#adapt parameters (filepaths) here first before running it --> creates png images from h5 file
python h5_to_img.py