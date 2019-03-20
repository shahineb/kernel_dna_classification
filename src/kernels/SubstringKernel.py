import os
import sys
import warnings
from functools import lru_cache
import numpy as np

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from src.kernels.Kernel import Kernel
from utils.decorators import accepts


class SubstringKernel(Kernel):
    """Implementation of Lodhi et al. 2002
    inspired from https://github.com/timshenkao/StringKernelSVM/blob/master/stringSVM.py
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

    @lru_cache(maxsize=64)
    def _Kprime(self, seq1, seq2, depth):
        if depth == 0:
            return 1
        elif min(len(seq1), len(seq2)) < depth:
            return 0
        else:
            part_sum = 0
            for j in range(1, len(seq2)):
                if seq2[j] == seq1[-1]:
                    part_sum += (self.decay_rate ** (len(seq2) - (j + 1) + 2)) * self._Kprime(seq1[:-1],
                                                                                              seq2[:j],
                                                                                              depth - 1)
            result = part_sum + self.decay_rate * self._Kprime(seq1[:-1],
                                                               seq2,
                                                               depth)
            return result

    @lru_cache(maxsize=64)
    def _evaluate(self, seq1, seq2):
        min_len = min(len(seq1), len(seq2))
        if min_len < self.n:
            return 0
        else:
            part_sum = 0
            for j in range(1, len(seq2)):
                if seq2[j] == seq1[-1]:
                    part_sum += self._Kprime(seq1[:-1], seq2[:j], self.n - 1)
            result = self(seq1[:-1], seq2) + self.decay_rate ** 2 * part_sum
            return result

    def _gram_matrix(self, X1, X2):
        len_X1 = len(X1)
        len_X2 = len(X2)
        gram_matrix = np.zeros((len_X1, len_X2), dtype=np.float32)

        seqs_norms = {1: dict(), 2: dict()}
        for i in range(len_X1):
            seqs_norms[1][i] = self._evaluate(X1[i], X1[i])
        for i in range(len_X2):
            seqs_norms[2][i] = self._evaluate(X2[i], X2[i])

        for i in range(len_X1):
            for j in range(len_X2):
                if X1[i] == X2[j]:
                    gram_matrix[i, j] = 1
                else:
                    gram_matrix[i, j] = self._evaluate(X1[i], X2[j]) / (seqs_norms[1][i] * seqs_norms[2][i]) ** 0.5
        return gram_matrix
