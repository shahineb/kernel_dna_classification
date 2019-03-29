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
        """Enforces binary labels to values to be in {-1, 1}

        Args:
            labels (np.ndarray): (n_sample,)
        """
        labels_ = labels.copy()
        labels_values = np.unique(labels)
        assert len(labels_values) == 2, "Please provide binary labels"
        labels_[labels == labels_values[0]] = -1
        labels_[labels == labels_values[1]] = 1
        return labels_

    def evaluate(self, y_true, y_pred, val=True):
        """Computes prediction accuracy

        Args:
            y_true (np.ndarray): true labels
            y_pred (np.ndarray): prediction
        """
        accuracy = np.mean(np.array(y_true == y_pred, dtype=int))
        set = 'Validation' if val else 'Training'
        if self.verbose:
            print('Accuracy on the {} set: {:.2f}'.format(set, accuracy))
        return accuracy

    @abstractmethod
    def fit(self, X, y, *args, **kwargs):
        """Fits classifier to data

        Args:
            X (np.ndarray)
            y (np.ndarray)
        """
        pass

    @abstractmethod
    @fitted
    def predict_prob(self, X):
        """Predicts proba of samples X

        Args:
            X (np.ndarray)
        """
        pass

    def predict(self, X, threshold=0.5):
        """Predicts label of samples X

        Args:
            X (np.ndarray)
            threshold (float): discrimination threshold
        """
        y_pred = np.array(self.predict_prob(X) >= threshold, dtype=int)
        y_pred[y_pred == 0] = -1
        return np.squeeze(y_pred)
