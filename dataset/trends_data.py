import torch
from torch.utils.data import Dataset

from sklearn.model_selection import ShuffleSplit

import pandas as pd

import numpy as np

import h5py

BUCKET_PATH = '/home/pattersonwu/apps/NeuroImaging/trends_dataset/'


def get_data():
    CSV_PATH = f'{BUCKET_PATH}/data'
    fnc_data = pd.read_csv(f"{CSV_PATH}/fnc.csv")
    sbm_data = pd.read_csv(f'{CSV_PATH}/loading.csv')
    fnc_data.set_index('Id', inplace=True)
    sbm_data.set_index('Id', inplace=True)
    return pd.concat([fnc_data, sbm_data], axis=1)

def get_targets():
    # image and mask directories
    CSV_PATH = f'{BUCKET_PATH}/data'
    train_data = pd.read_csv(f'{CSV_PATH}/train_scores.csv')
    return train_data


def load_subject(file_name):
    MRI_PATH = f'/home/pattersonwu/apps/NeuroImaging/trends_dataset/fMRI_train'
    subject_data = h5py.File(f'{MRI_PATH}/{file_name}.mat', 'r')['SM_feature']
    subject_data = np.transpose(subject_data[()], (3, 2, 1, 0))
    return subject_data

# def detrend(data, axis=-1, type='linear', inplace=False):
#     return scipy.signal.detrend(data, axis, type, overwrite_data=inplace)
#
#
# def normalize(data, norm='l2', axis='1', inplace='false', return_norm=True):
#     return sklearn.preprocessing.normalize(data, norm, axis, return_norm, copy=inplace)


# def select_features(X, y, K):
#     result = []
#     selector = SelectKBest(score_func=f_regression, k=K)
#     for col in y.columns:
#         selector.fit_transform(X, y[col])
#         columns_selected = selector.get_support(indices=True)
#         df = X.iloc[:, columns_selected]
#         result.append(df)
#     X_new = reduce(lambda x, y: pd.concat((x, y[y.columns.difference(x.columns)]), axis=1), result)
#     return X_new

# We are dynamically loading fnc due to the number of columns

class TrendsDataset(Dataset):
    def __init__(self, mode='train'):
        targets = get_targets()
        data = get_data()
        targets.set_index('Id', inplace=True)
        targets.dropna(axis=0, inplace=True)
        ss = ShuffleSplit(n_splits=1, test_size=0.2, random_state=1227)
        ids = targets.index
        for train_index, valid_index in ss.split(ids):
            train_ids, valid_ids = ids[train_index], ids[valid_index]
        if mode == 'train':
            self.data = data.loc[train_ids]
            self.targets = targets.loc[train_ids]
            self.ids = train_ids
        elif mode == 'validation':
            self.data = data.loc[valid_ids]
            self.targets = targets.loc[valid_ids]
            self.ids = valid_ids
        del data, targets

    def __getitem__(self, index):
        id = self.ids[index]
        data_tensor = torch.tensor(self.data.loc[id], dtype=torch.float)
        targets = torch.tensor(self.targets.loc[id], dtype=torch.float)
        return {'data': data_tensor, 'targets': targets}

    def __len__(self):
        return len(self.ids)


class MRIDataset(Dataset):
    def __init__(self, mode='train'):
        targets = get_targets()
        targets.set_index('Id', inplace=True)
        data = get_data()
        targets.dropna(axis=0, inplace=True)
        # Train validation split using KFold
        ids = targets.index
        ss = ShuffleSplit(n_splits=1, test_size=0.2, random_state=1227)
        for train_index, valid_index in ss.split(ids):
            self.train_index, self.valid_index = ids[train_index], ids[valid_index]
        if mode == 'train':
            subject_data = targets.loc[self.train_index]
            self.data = data.loc[self.train_index]
            self.targets = targets.loc[self.train_index]
        elif mode == 'validation':
            subject_data = targets.loc[self.valid_index]
            self.data = data.loc[self.valid_index]
            self.targets = targets.loc[self.valid_index]

        self.ids = subject_data.index

    def __getitem__(self, index):
        id = self.ids[index]
        subject_data = load_subject(id)
        targets = torch.tensor(self.targets.loc[id], dtype=torch.float)
        data = torch.tensor(self.data.loc[id], dtype=torch.float)
        scans = torch.tensor(subject_data, dtype=torch.float)
        del subject_data
        return {'scans': scans, 'data': data, 'targets': targets}

    def __len__(self):
        return len(self.ids)