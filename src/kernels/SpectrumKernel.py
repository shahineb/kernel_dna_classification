import os
import sys
import warnings
from six import string_types
from itertools import product
from joblib import Parallel, delayed
import numpy as np

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from src.kernels.Kernel import Kernel
from utils.decorators import accepts
import config


class SpectrumKernel(Kernel):
    """Implementation of Leslie et al., 2002
    with strings preindexation
    """
    NMAX = 6

    @accepts(int, string_types, int)
    def __init__(self, n, charset, verbose=0):
        """
        Args:
            n (int): n-uplet size to consider
            charset (str): charset for preindexation (typically "ATCG")
        """
        super(SpectrumKernel, self).__init__(verbose)
        self._n = n
        self._charset = charset
        product_seed = self._n * ("ATCG",)
        patterns = product(*product_seed)
        join = lambda x: "".join(x)
        if self._n > SpectrumKernel.NMAX:
            warnings.warn(f"Becomes computationally expensive when n > {SpectrumKernel.NMAX}")
            self._patterns = Parallel(n_jobs=config.n_jobs)(delayed(join)(pattern) for pattern in patterns)
        else:
            self._patterns = list(map(join, patterns))

    @property
    def n(self):
        return self._n

    @property
    def charset(self):
        return self._charset

    @property
    def patterns(self):
        return self._patterns

    def _get_tuple(self, seq, position):
        try:
            return seq[position:position + self.n]
        except IndexError:
            raise IndexError("Position out of range for tuple")

    def _count_pattern(self, pattern, count_dict):
        count_dict[pattern] += 1
        return count_dict

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
            counts1 = {pattern: 0 for pattern in self.patterns}
            counts2 = {pattern: 0 for pattern in self.patterns}

            for i in range(max_len - self.n):
                try:
                    subseq1 = self._get_tuple(seq1, i)
                    counts1 = self._count_pattern(subseq1, counts1)
                except KeyError:
                    pass
                try:
                    subseq2 = self._get_tuple(seq2, i)
                    counts2 = self._count_pattern(subseq2, counts2)
                except KeyError:
                    continue

            feats1 = np.fromiter(counts1.values(), dtype=np.float32)
            feats2 = np.fromiter(counts2.values(), dtype=np.float32)
            return np.inner(feats1, feats2)

    def _pairwise(self, X1, X2):
        """
        Args:
            X1 (np.ndarray)
            X2 (np.ndarray)
        """
        min_len = min(map(len, np.hstack([X1, X2])))
        max_len = max(map(len, np.hstack([X1, X2])))
        if min_len < self.n:
            return 0
        else:
            counts1 = {idx: {perm: 0 for perm in self.patterns} for idx in range(len(X1))}
            counts2 = {idx: {perm: 0 for perm in self.patterns} for idx in range(len(X2))}
            for i in range(max_len - self.n):
                for idx, seq in enumerate(X1):
                    try:
                        subseq = self._get_tuple(seq, i)
                        counts1[idx] = self._count_pattern(subseq, counts1[idx])
                    except KeyError:
                        pass
                for idx, seq in enumerate(X2):
                    try:
                        subseq = self._get_tuple(seq, i)
                        counts2[idx] = self._count_pattern(subseq, counts2[idx])
                    except KeyError:
                        pass

            feats1 = np.array([np.fromiter(foo.values(), dtype=np.float32) for foo in counts1.values()])
            norms1 = np.linalg.norm(feats1, axis=1).reshape(-1, 1)
            feats2 = np.array([np.fromiter(foo.values(), dtype=np.float32) for foo in counts2.values()])
            norms2 = np.linalg.norm(feats2, axis=1).reshape(-1, 1)
            return np.inner(feats1 / norms1, feats2 / norms2)
