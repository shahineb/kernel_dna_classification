import os
import sys
import warnings
from six import string_types
from itertools import permutations
from numba import vectorize
import numpy as np

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from src.kernels.Kernel import Kernel
from utils.decorators import accepts


class SpectrumKernel(Kernel):
    """Implementation of Leslie et al., 2002
    with strings preindexation
    """
    NMAX = 6

    @accepts(int, string_types)
    def __init__(self, n, charset):
        """
        Args:
            n (int): n-uplet size to consider
            charset (str): charset for preindexation (typically "ATCG")
        """
        self._n = n
        if self._n > SpectrumKernel.NMAX:
            warnings.warn(f"Becomes computationally expensive when n > {SpectrumKernel.NMAX}")
        self._charset = charset
        permutation_seed = (2 + max(n - len(charset), 0)) * charset
        helper = lambda x: "".join(x)
        self._char_permutations = list(map(helper, set(permutations(permutation_seed, self._n))))

    @property
    def n(self):
        return self._n

    @property
    def charset(self):
        return self._charset

    @property
    def char_permutations(self):
        return self._char_permutations

    @accepts(string_types, int)
    def _get_tuple(self, seq, position):
        try:
            return seq[position:position + self.n]
        except IndexError:
            raise IndexError("Position out of range for tuple")

    @accepts(string_types, string_types)
    def _evaluate(self, seq1, seq2):
        """
        Args:
            seq1 (str): dna sequence
            seq2 (str): dna sequence
        """
        min_len = min(len(seq1), len(seq2))
        if min_len < self.n:
            return 0
        else:
            max_len = max(len(seq1), len(seq2))
            counts1 = {perm: 0 for perm in self.char_permutations}
            counts2 = {perm: 0 for perm in self.char_permutations}

            for i in range(max_len - self.n):
                try:
                    subseq1 = self._get_tuple(seq1, i)
                    counts1[subseq1] += 1
                except KeyError:
                    pass
                try:
                    subseq2 = self._get_tuple(seq2, i)
                    counts2[subseq2] += 1
                except KeyError:
                    continue

            feats1 = np.fromiter(counts1.values(), dtype=np.float32)
            feats2 = np.fromiter(counts2.values(), dtype=np.float32)
            return np.inner(feats1, feats2)

    @accepts(np.ndarray, np.ndarray)
    def _gram_matrix(self, X1, X2):
        min_len = min(map(len, np.hstack([X1, X2])))
        max_len = max(map(len, np.hstack([X1, X2])))
        if min_len < self.n:
            return 0
        else:
            counts1 = {idx: {perm: 0 for perm in self.char_permutations} for idx in range(len(X1))}
            counts2 = {idx: {perm: 0 for perm in self.char_permutations} for idx in range(len(X2))}
            for i in range(max_len - self.n):
                for idx, seq in enumerate(X1):
                    try:
                        subseq = self._get_tuple(seq, i)
                        counts1[idx][subseq] += 1
                    except KeyError:
                        pass
                for idx, seq in enumerate(X2):
                    try:
                        subseq = self._get_tuple(seq, i)
                        counts2[idx][subseq] += 1
                    except KeyError:
                        pass

            feats1 = np.array([np.fromiter(foo.values(), dtype=np.float32) for foo in counts1.values()])
            feats2 = np.array([np.fromiter(foo.values(), dtype=np.float32) for foo in counts2.values()])
            return np.inner(feats1, feats2)
