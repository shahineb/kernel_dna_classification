import os
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
