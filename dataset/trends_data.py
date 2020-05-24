import sys
import os

import scipy.signal
import sklearn.preprocessing
from sklearn.feature_selection import SelectKBest, f_regression
from functools import reduce
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

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
    MRI_PATH = f'/home/pattersonwu/NeuroImaging/train/fMRI_train'
    subject_data = h5py.File(f'{MRI_PATH}/{file_name}.mat', 'r')['SM_feature']
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
    subject_data = load_subject(X.index[200])
    X_data = X.loc[X.index[200]]
    print(subject_data.shape)
    print(X_data.shape)

    # print('Sample submission')
    # sample = pd.read_csv(f"{BUCKET_PATH}/sample_submission.csv")
    # print(sample.shape)


class TrendsDataset(Dataset):
    def __init__(self):
        loading_data, train_data, fnc_data = load_dataset()
        data1 = join_dataset(loading_data, train_data, fnc_data)
        data1 = drop_na(data1)
        X, self.Y = data_split(data1)
        self.X = select_features(X, self.Y, 100)
        self.ids = X.index

    def __getitem__(self, index):
        id = self.ids[index]
        subject_data = load_subject(id)
        start = time.time()
        scans_transposed = np.transpose(subject_data[()], (3,2,1,0))
        del subject_data
        scans = torch.tensor(scans_transposed, dtype=torch.float)
        del scans_transposed
        end = time.time()
        print("Time to turn to tensor: " + str(end - start))
        X = torch.tensor(self.X.loc[id], dtype=torch.float)
        targets = torch.tensor(self.Y.loc[id], dtype=torch.float)
        return [scans, X, targets]

    def __len__(self):
        return len(self.ids)
