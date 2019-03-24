import os
import sys
import numpy as np
from abc import ABCMeta, abstractmethod

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from utils.decorators import fitted


class Classifier:
    """General abstract class for Classifier
    """
    __metaclass__ = ABCMeta

    def __init__(self, kernel, verbose, *args, **kwargs):
        """
        Args:
            verbose (int): in {0, 1}
        """
        self._kernel = kernel
        self._verbose = bool(verbose)
        self._fitted = False
        self._Xtr = None

    @property
    def kernel(self):
        return self._kernel

    @property
    @fitted
    def Xtr(self):
        return self._Xtr

    @property
    def verbose(self):
        return self._verbose

    @staticmethod
    def format_binary_labels(labels):
        labels_ = labels.copy()
        labels_values = np.unique(labels)
        assert len(labels_values) == 2, "Please provide binary labels"
        labels_[labels == labels_values[0]] = -1
        labels_[labels == labels_values[1]] = 1
        return labels_

    @staticmethod
    def evaluate(y_true, y_pred, val=True):
        accuracy = np.mean(np.array(y_true == y_pred, dtype=int))
        set = 'Validation' if val else 'Training'
        print('Accuracy on the {} set: {:.2f}'.format(set, accuracy))


    @abstractmethod
    def fit(self, x, y, *args, **kwargs):
        """Evaluates kernel on samples x and y

        Args:
            x (hashable)
            y (hashable)
        """
        pass

    @abstractmethod
    @fitted
    def predict_prob(self, X):
        """Predicts proba on samples x

        Args:

        """
        pass

    def predict(self, X, threshold=0.5):
        y_pred = np.array(self.predict_prob(X) >= threshold, dtype=int)
        y_pred[y_pred == 0] = -1
<<<<<<< .merge_file_L3vk3p
        return np.squeeze(y_pred)

=======
        return y_pred
>>>>>>> .merge_file_jdcbpO
