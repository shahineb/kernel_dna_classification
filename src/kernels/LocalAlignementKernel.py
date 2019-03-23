import os
import sys
import numpy as np
import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()})

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from src.kernels.Kernel import Kernel
from utils.decorators import accepts
import src.kernels.bin.localalignement as c


class LocalAlignementKernel(Kernel):
    """Implementation of Vert et al., 2004
    for affine gap penalty function
    """

    @accepts(np.ndarray, dict, float, float, float)
    def __init__(self, S, char2idx, e, d, beta):
        """
        Args:
            S (np.ndarray): substitution matrix
            e (float): affine gap penalty slope
            d (float): affine gap penalty intercept
            beta (float): local alignement parameter
        """
        self._S = S
        self._e = e
        self._d = d
        self._beta = beta
        self._char2idx = char2idx

    @property
    def S(self):
        return self._S

    @property
    def e(self):
        return self._e

    @property
    def d(self):
        return self._d

    @property
    def beta(self):
        return self._beta

    @property
    def char2idx(self):
        return self._char2idx

    def _evaluate(self, seq1, seq2):
        return c._evaluate(seq1, seq2, self.S, self.char2idx, self.e, self.d, self.beta)
