from numpy import ndarray
from six import string_types
from abc import ABCMeta, abstractmethod


class Kernel:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, x, y):
        if isinstance(x, ndarray) and isinstance(y, ndarray):
            return self._gram_matrix(x, y)
        elif isinstance(x, string_types) and isinstance(y, string_types):
            return self._evaluate(x, y)
        else:
            raise TypeError("Not implemented for specified input type")

    @abstractmethod
    def _gram_matrix(self, x, y):
        pass

    @abstractmethod
    def _evaluate(self, x, y):
        pass
