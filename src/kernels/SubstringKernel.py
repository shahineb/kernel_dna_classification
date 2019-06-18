import os
import sys
import warnings
from functools import lru_cache
import numpy as np
import pyximport
from .Kernel import StringKernel

pyximport.install(setup_args={"include_dirs": np.get_include()})

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

import src.kernels.bin.substring as c
from utils.decorators import accepts


class SubstringKernel(StringKernel):
    """Implementation of Lodhi et al. 2002
    inspired from https://github.com/timshenkao/StringKernelSVM/blob/master/stringSVM.py

    Implementation is optimized with cache usage for recursive computations and
    cython static types declaration

    Attributes:
        n (int): n-mer size to consider
        decay_rate (float): decay parameter in ]0, 1[
    """
    NMAX = 2

    @accepts(int, float, int)
    def __init__(self, n, decay_rate, verbose=1):
        super(SubstringKernel, self).__init__(n, verbose)
        if self._n > SubstringKernel.NMAX:
            warnings.warn(f"Becomes computationally expensive when n > {SubstringKernel.NMAX}")
        self._decay_rate = decay_rate

    @property
    def decay_rate(self):
        return self._decay_rate

    @lru_cache(maxsize=64)
    def _Kprime(self, seq1, seq2, depth):
        return c._Kprime(seq1, seq2, depth, self.n, self.decay_rate)

    @lru_cache(maxsize=64)
    def _evaluate(self, seq1, seq2):
        return c._evaluate(seq1, seq2, self.n, self.decay_rate)
