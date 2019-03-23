import numpy as np
from six import string_types
from abc import ABCMeta, abstractmethod


class Kernel:
    """General abstract class for Kernel
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, x, y):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            return self._gram_matrix(x, y)
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

    def _gram_matrix(self, X1, X2):
        """Computes Gram Matrix induced by kernel for datasets x and y

        Args:
            X1 (np.ndarray)
            X2 (np.ndarray)
        """
        X = [X1, X2]
        X.sort(key=len)
        min_X, min_len = X[0], len(X[0])
        max_X, max_len = X[1], len(X[1])
        gram_matrix = np.zeros((min_len, max_len), dtype=np.float32)

        seqs_norms = {0: dict(), 1: dict()}
        for i in range(min_len):
            buffer = self._evaluate(min_X[i], min_X[i])
            seqs_norms[0][i] = buffer
            if min_X[i] == max_X[i]:
                seqs_norms[1][i] = buffer
            else:
                seqs_norms[1][i] = self._evaluate(max_X[i], max_X[i])
        for i in range(min_len, max_len):
            seqs_norms[1][i] = self._evaluate(max_X[i], max_X[i])
        for i in range(min_len):
            for j in range(max_len):
                if min_X[i] == max_X[j]:
                    gram_matrix[i, j] = 1
                else:
                    gram_matrix[i, j] = self._evaluate(min_X[i], max_X[j]) / (seqs_norms[0][i] * seqs_norms[1][j]) ** 0.5
        return gram_matrix
