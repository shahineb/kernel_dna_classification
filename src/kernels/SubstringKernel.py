import os
import sys
import warnings
from functools import lru_cache
from six import string_types

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from src.kernels.Kernel import Kernel
from utils.decorators import accepts


class SubstringKernel(Kernel):
    NMAX = 2

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
    def __call__(self, seq1, seq2):
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
