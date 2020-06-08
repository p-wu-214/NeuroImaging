import sys
import torch

from torch.utils.data import Dataset
import torch_xla.core.xla_model as xm

import scipy.signal
import sklearn.preprocessing
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import ShuffleSplit
from functools import reduce
import pandas as pd
import numpy as np

import h5py

import time

BUCKET_PATH = 'gs://trends_dataset'


def load_dataset():
    # image and mask directories
    CSV_PATH = f'{BUCKET_PATH}/data'
    train_data_dir = f'{BUCKET_PATH}/fMRI_train'
    print('Reading data...')
    loading_data = pd.read_csv(f'{CSV_PATH}/loading.csv')
    train_data = pd.read_csv(f'{CSV_PATH}/train_scores.csv')
    fnc_data = pd.read_csv(f"{CSV_PATH}/fnc.csv")
    print('Finish reading data...')
    return loading_data, train_data, fnc_data


# Detrending over time for all voxels
def detrend(data, axis=-1, type='linear', inplace=False):
    return scipy.signal.detrend(data, axis, type, overwrite_data=inplace)


def normalize(data, norm='l2', axis='1', inplace='false', return_norm=True):
    return sklearn.preprocessing.normalize(data, norm, axis, return_norm, copy=inplace)


def display_data(data1, data2, data3):
    print(data1.head())
    print(data1.describe())
    print(data2.head())
    print(data2.describe())
    print(data3.head())
    print(data3.describe())


def get_nan(data):
    # displays NaNs in X
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum() / data.isnull().count() * 100).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])


def join_dataset(data1, data2, data3):
    result = data2.join(data1.set_index('Id'), on='Id')
    result = result.join(data3.set_index('Id'), on='Id')
    return result


def drop_na(data):
    data = data.dropna(axis=0)
    data = data.set_index('Id')
    return data


def data_split(data):
    X = data.drop(['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2'], axis=1)
    Y = data[['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']]
    return X, Y


def select_features(X, y, K):
    result = []
    selector = SelectKBest(score_func=f_regression, k=K)
    for col in y.columns:
        selector.fit_transform(X, y[col])
        columns_selected = selector.get_support(indices=True)
        df = X.iloc[:, columns_selected]
        result.append(df)
    X_new = reduce(lambda x, y: pd.concat((x, y[y.columns.difference(x.columns)]), axis=1), result)
    return X_new


def load_subject(file_name):
    MRI_PATH = f'/home/pattersonwu/apps/NeuroImaging/trends_dataset/fMRI_train'
    subject_data = h5py.File(f'{MRI_PATH}/{file_name}.mat', 'r')['SM_feature']
    subject_data = np.transpose(subject_data[()], (3, 2, 1, 0))
    return subject_data


if __name__ == '__main__':
    print('Staring program')
    loading_data, train_data, fnc_data = load_dataset()
    # display_data(loading_data, train_data, fnc_data)

    to_select_feature = join_dataset(loading_data, train_data, fnc_data)
    to_select_feature = drop_na(to_select_feature)
    print('Dataset all joined')
    print(to_select_feature.head())

    X, Y = data_split(to_select_feature)
    print('X without feature select')
    print(X.head())
    print('Y without feature select')
    print(Y.head())

    X = select_features(X, Y, 100)
    print('After reduce')
    print(X)

    print('Reading in fMRI scan data')
    subject_data = load_subject('10001')
    # X_data = X.loc[X.index[200]]
    print(subject_data.shape)
    # print(X_data.shape)
    print(subject_data.min(), subject_data.max(), subject_data.mean())

    # print('Sample submission')
    # sample = pd.read_csv(f"{BUCKET_PATH}/sample_submission.csv")
    # print(sample.shape)


class TrendsDataset(Dataset):
    def __init__(self, mode='train'):
        loading_data, train_data, fnc_data = load_dataset()
        samples = join_dataset(loading_data, train_data, fnc_data)
        samples = drop_na(samples)

        # Train validation split using KFold
        ids = samples.index
        ss = ShuffleSplit(n_splits=1, test_size=0.2, random_state=1227)
        for train_index, valid_index in ss.split(ids):
            self.train_index, self.valid_index = ids[train_index], ids[valid_index]
        if mode == 'train':
            print('Train')
            data = samples.loc[self.train_index]
        elif mode == 'validation':
            print('Validation')
            data = samples.loc[self.valid_index]
        del samples

        X, Y = data_split(data)
        X = select_features(X, Y, 100)
        self.ids = data.index
        self.X = torch.tensor(X.to_numpy(), dtype=torch.float)
        self.Y = torch.tensor(Y.to_numpy(), dtype=torch.float)
        del X, Y
        print(f'{mode} dataset with X: {self.X.shape} and Y: {self.Y.shape}')

    def __getitem__(self, index):
        subject_data = load_subject(self.ids[index])
        X = self.X[index]
        targets = self.Y[index]
        scans = torch.tensor(subject_data, dtype=torch.float)
        del subject_data
        return {'scans': scans, 'X': X, 'targets': targets}

    def __len__(self):
        return len(self.ids)
