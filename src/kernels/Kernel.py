import os
import sys
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from six import string_types
from abc import ABCMeta, abstractmethod

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

import config


class Kernel:
    """General abstract class for Kernel
    """
    __metaclass__ = ABCMeta

    def __init__(self, verbose, *args, **kwargs):
        """
        Args:
            verbose (int): in {0, 1}
        """
        self._verbose = not verbose

    @property
    def verbose(self):
        return self._verbose

    def __call__(self, x, y):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            return self._pairwise(x, y)
        elif isinstance(x, string_types) and isinstance(y, string_types):
            return self._evaluate(x, y)
        else:
            raise TypeError("Not implemented for specified input type")

    @abstractmethod
    def _evaluate(self, x, y):
        """Evaluates kernel on samples x and y

        Args:
            x (hashable)
            y (hashable)
        """
        pass

    def _pairwise(self, X1, X2):
        """Computes normalized pairwise terms matric induced by kernel for datasets x and y
        Norms are precomputed and stored so that they can be called for normalization
        We make use of symmetry to avoid recomputing terms

        Args:
            X1 (np.ndarray)
            X2 (np.ndarray)
        """
        # Order arrays by dimensionality and initialize matrix
        X = [X1, X2]
        X.sort(key=len)
        min_X, min_len = X[0], len(X[0])
        max_X, max_len = X[1], len(X[1])
        pairwise_matrix = np.zeros((min_len, max_len), dtype=np.float32)

        # Precompute all norms K(x_i,x_i)
        min_norms = Parallel(n_jobs=config.n_jobs)(delayed(self._evaluate)(x, x) for x in min_X)

        copy_or_eval = lambda i: min_norms[i] if min_X[i] == max_X[i] else self._evaluate(max_X[i], max_X[i])
        max_norms = Parallel(n_jobs=config.n_jobs)(delayed(copy_or_eval)(i) for i in range(min_len))
        leftovers = Parallel(n_jobs=config.n_jobs)(delayed(self._evaluate)(x, x) for x in max_X[min_len:])
        max_norms = max_norms + leftovers

        norms = {0: {i: value for (i, value) in enumerate(min_norms)},
                 1: {i: value for (i, value) in enumerate(max_norms)}}

        # Compute all normalized inner products K(x_i, x_j) / ( K(x_i, x_i) *  K(x_j, x_j)) ** 0.5
        def helper(i):
            gram_row_i = np.zeros(max_len)
            for j in range(i, max_len):
                if min_X[i] == max_X[j]:
                    gram_row_i[j] = 1
                else:
                    gram_row_i[j] = self._evaluate(min_X[i], max_X[j]) / np.sqrt(norms[0][i] * norms[1][j])
            return gram_row_i

        rows = Parallel(n_jobs=config.n_jobs)(delayed(helper)(i) for i in tqdm(range(min_len), disable=self.verbose))
        pairwise_matrix = np.stack(rows)
        pairwise_matrix[:, :min_len] = pairwise_matrix[:, :min_len] + pairwise_matrix.T[:min_len] - np.diag(pairwise_matrix) * np.eye(min_len)

        # Return matrix in correct orientation
        if min_len == len(X2):
            return pairwise_matrix.transpose()
        else:
            return pairwise_matrix
