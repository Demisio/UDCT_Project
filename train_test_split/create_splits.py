import numpy as np
import os
from sklearn import model_selection
import pandas as pd

split_path = './Heart/heart_ids.txt'
df_ids = pd.read_csv('heart_ids.txt', header=None)
df_labs = pd.read_csv('heart_gt_ids.txt', header=None)

n_splits = 5
# kfold = model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=20)

kfold = model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=20)
i = 1
for train, test in kfold.split(df_ids, df_labs):
    train_ids = ids[train]
    test_ids = ids[test]

    with open(os.path.join(splitpath, label, 'train_fold_{}.txt'.format(i)), 'w') as file:
        for el in train_ids:
            file.write(str(el)+'\n')
    with open(os.path.join(splitpath, label, 'test_fold_{}.txt'.format(i)), 'w') as file:
        for el in test_ids:
            file.write(str(el)+'\n')
    i += 1
