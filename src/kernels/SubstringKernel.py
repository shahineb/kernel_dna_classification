import os
import sys
import warnings
from functools import lru_cache
import numpy as np
import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()})

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from src.kernels.Kernel import Kernel
from utils.decorators import accepts
import src.kernels.bin.substringkernel as c


class SubstringKernel(Kernel):
    """Implementation of Lodhi et al. 2002
    inspired from https://github.com/timshenkao/StringKernelSVM/blob/master/stringSVM.py

    Attributes:
        n (int): n-uplet size to consider
        decay_rate (float): decay parameter in ]0, 1[
    """
    NMAX = 2

    @accepts(int, float)
    def __init__(self, n, decay_rate):
        self._n = n
        if self._n > SubstringKernel.NMAX:
            warnings.warn(f"Becomes computationally expensive when n > {SubstringKernel.NMAX}")
        self._decay_rate = decay_rate

    @property
    def n(self):
        return self._n

    @property
    def decay_rate(self):
        return self._decay_rate

    @lru_cache(maxsize=32)
    def _Kprime(self, seq1, seq2, depth):
        return c._Kprime(seq1, seq2, depth, self.n, self.decay_rate)

    @lru_cache(maxsize=32)
    def _evaluate(self, seq1, seq2):
        return c._evaluate(seq1, seq2, self.n, self.decay_rate)

    def _gram_matrix(self, X1, X2):
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
