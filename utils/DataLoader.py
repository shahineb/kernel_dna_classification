import os
import numpy as np
import pandas as pd

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")


class DataLoader:
    """small loading utility
    """

    data_dirname = "data"

    def __init__(self, cache=False):
        self._data_dir = os.path.join(base_dir, DataLoader.data_dirname)

    @property
    def data_dir(self):
        return self._data_dir

    def load(self, dataset_id, as_array=False):
        # TODO : efficient load by id only and add label option
        df = pd.read_csv(os.path.join(self.data_dir, dataset_id), index_col=0)
        return df.values.squeeze() if as_array else df

    @staticmethod
    def train_val_split(x, y, val_size=0.33, random_state=42):
        assert x.shape[0] == y.shape[0]
        length = x.shape[0]
        rng = np.random.RandomState(random_state)
        idx = rng.permutation(length)
        x = x[idx]
        y = y[idx]
        milestone = int(length*(1-val_size))
        x_train, y_train = x[:milestone], y[:milestone]
        x_val, y_val = x[milestone:], y[milestone:]
        return x_train, x_val, y_train, y_val

    def get_train_val(self, k=0, val_size=0.33, random_state=42):
        x_train = self.load('Xtr{}.csv'.format(k)).values
        y_train = self.load('Ytr{}.csv'.format(k)).values
        x_train, y_train = np.squeeze(x_train), np.squeeze(y_train)
        y_train[y_train == 0] = -1
        return self.train_val_split(x_train, y_train, val_size=val_size, random_state=random_state)

    def get_test(self, k=0):
        x_test = self.load('Xte{}.csv'.format(k)).values
        return np.squeeze(x_test)