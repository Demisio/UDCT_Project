import h5py
import numpy as np
import os
from sklearn import model_selection
import pandas as pd


n_splits = 10
kfold = model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=20)

i = 1
for train, test in kfold.split(ids, labs):
    train_ids = ids[train]
    test_ids = ids[test]

    with open(os.path.join(splitpath, label, 'train_fold_{}.txt'.format(i)), 'w') as file:
        for el in train_ids:
            file.write(str(el)+'\n')
    with open(os.path.join(splitpath, label, 'test_fold_{}.txt'.format(i)), 'w') as file:
        for el in test_ids:
            file.write(str(el)+'\n')
    i += 1