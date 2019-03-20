from abc import ABCMeta, abstractmethod


class Kernel:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, x, y):
        pass
