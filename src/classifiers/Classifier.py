import os
import sys
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from six import string_types
from abc import ABCMeta, abstractmethod

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

class Classifier:
    """General abstract class for Classifier
    """
    __metaclass__ = ABCMeta

    def __init__(self, verbose, *args, **kwargs):
        """
        Args:
            verbose (int): in {0, 1}
        """
        self._verbose = bool(verbose)

    @property
    def verbose(self):
        return self._verbose

    @staticmethod
    def evaluate(y_true, y_pred, val=False):
        #import pdb; pdb.set_trace()
        array = np.array(y_true == y_pred, dtype=int)
        array[array == -1] = 0
        accuracy = np.mean(array)
        set = 'Validation' if val else 'Training'
        print('Accuracy on the {} set: {:.2f}'.format(set, accuracy))


    @abstractmethod
    def fit(self, x, y):
        """Evaluates kernel on samples x and y

        Args:
            x (hashable)
            y (hashable)
        """
        pass

    @abstractmethod
    def predict_prob(self, X):
        """Predicts proba on samples x

        Args:

        """
        pass

    def predict(self, X, threshold=0.5):
        y_pred = np.array(self.predict_prob(X) >= threshold, dtype=int)
        y_pred[y_pred == 0] = -1
        return np.squeeze(y_pred)

