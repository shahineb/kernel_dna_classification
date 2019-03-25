import os
import sys
import numpy as np
from scipy.spatial.distance import cdist

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from src.kernels.Kernel import Kernel
from utils.decorators import accepts


class GaussianKernel(Kernel):
    """Implementation of Gaussian kernel
    """

    @accepts(float)
    def __init__(self, std=1.0, verbose=1):
        """
        Args:
            std (float): standard deviation
        """
        super(GaussianKernel, self).__init__(verbose)
        self._std = std

    @property
    def std(self):
        return self._std

    def _evaluate(self, x1, x2):
        """
        Args:
            x1 (array): vector
            x2 (array): vector
        """
        return np.exp(-(x1 - x2)**2 / (2 * self._std))

    def _gram_matrix(self, X1, X2):
        """
        Args:
            X1 (np.ndarray)
            X2 (np.ndarray)
        """
        pairwise_dists = cdist(X1, X2, 'euclidean')
        return np.exp(-pairwise_dists ** 2 / self._std ** 2)
