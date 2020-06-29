import torch
from torch.utils.data import Dataset

from sklearn.model_selection import ShuffleSplit

import pandas as pd

import numpy as np

import h5py

BUCKET_PATH = '/home/pattersonwu/apps/NeuroImaging/trends_dataset/'

def get_sbm():
    # image and mask directories
    CSV_PATH = f'{BUCKET_PATH}/data'
    loading_data = pd.read_csv(f'{CSV_PATH}/loading.csv').astype(np.float32)
    return loading_data


def get_fnc():
    CSV_PATH = f'{BUCKET_PATH}/data'
    fnc_data = pd.read_csv(f"{CSV_PATH}/fnc.csv").astype(np.float32)
    return fnc_data

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

# class TrendsDataset(Dataset):
#     def __init__(self, mode='train'):
#         sbm = get_sbm()
#         targets = get_targets()
#         fnc = get_fnc()
#         targets.set_index('Id', inplace=True)
#         sbm.set_index('Id', inplace=True)
#         fnc.set_index('Id', inplace=True)
#         targets.dropna(axis=0, inplace=True)
#         ss = ShuffleSplit(n_splits=1, test_size=0.2, random_state=1227)
#         ids = targets.index
#         for train_index, valid_index in ss.split(ids):
#             train_ids, valid_ids = ids[train_index], ids[valid_index]
#         if mode == 'train':
#             print('Train')
#             self.sbm = sbm.loc[train_ids]
#             self.fnc = fnc.loc[train_ids]
#             self.targets = targets.loc[train_ids]
#             self.ids = train_ids
#         elif mode == 'validation':
#             print('Validation')
#             self.sbm = sbm.loc[valid_ids]
#             self.fnc = fnc.loc[train_ids]
#             self.targets = targets.loc[valid_ids]
#             self.ids = valid_ids
#         del sbm, targets
#
#     def __getitem__(self, index):
#         id = self.ids[index]
#         sbm_tensor = torch.tensor(self.sbm.loc[id], dtype=torch.float)
#         fnc_tensor = torch.tensor(self.fnc.loc[id], dtype=torch.float)
#         targets = torch.tensor(self.targets.loc[id], dtype=torch.float)
#         return {'sbm_data' : sbm_tensor, 'fnc_data' : fnc_tensor, 'targets' : targets}
#
#     def __len__(self):
#         return len(self.ids)


class MRIDataset(Dataset):
    def __init__(self, mode='train'):
        targets = get_targets()
        sbm = get_sbm()
        fnc = get_fnc()
        targets.set_index('Id', inplace=True)
        sbm.set_index('Id', inplace=True)
        sbm.fillna(0)
        fnc.set_index('Id', inplace=True)
        fnc.fillna(0)
        targets.dropna(axis=0, inplace=True)
        # Train validation split using KFold
        ids = targets.index
        ss = ShuffleSplit(n_splits=1, test_size=0.2, random_state=1227)
        for train_index, valid_index in ss.split(ids):
            self.train_index, self.valid_index = ids[train_index], ids[valid_index]
        if mode == 'train':
            data = targets.loc[self.train_index]
            self.sbm = sbm.loc[self.train_index]
            self.fnc = fnc.loc[self.train_index]
            self.targets = targets.loc[self.train_index]
        elif mode == 'validation':
            data = targets.loc[self.valid_index]
            self.sbm = sbm.loc[self.valid_index]
            self.fnc = fnc.loc[self.valid_index]
            self.targets = targets.loc[self.valid_index]

        self.ids = data.index

    def __getitem__(self, index):
        id = self.ids[index]
        subject_data = load_subject(id)
        targets = torch.tensor(self.targets.loc[id], dtype=torch.float32)
        sbm_tensor = torch.tensor(self.sbm.loc[id], dtype=torch.float32)
        fnc_tensor = torch.tensor(self.fnc.loc[id], dtype=torch.float32)
        scans = torch.tensor(subject_data, dtype=torch.float)
        del subject_data
        return {'scans': scans, 'sbm': sbm_tensor, 'fnc': fnc_tensor, 'targets': targets}

    def __len__(self):
        return len(self.ids)