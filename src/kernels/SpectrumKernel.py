import os
import sys
import warnings
from six import string_types
from itertools import product
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from src.kernels.Kernel import StringKernel
from utils.decorators import accepts
import config


class SpectrumKernel(StringKernel):
    """Implementation of Leslie et al., 2002
    with strings preindexation
    """
    NMAX = 8

    @accepts(int, string_types, int)
    def __init__(self, n, charset, verbose=0):
        """
        Args:
            n (int): n-mers size to consider
            charset (str): charset for preindexation (typically "ATCG")
            verbose (int): {0, 1}
        """
        super(SpectrumKernel, self).__init__(n, verbose)
        self._n = n
        self._charset = charset
        # Generate all possible patterns of size n
        product_seed = self._n * ("ATCG",)
        patterns = product(*product_seed)
        join = lambda x: "".join(x)
        if self._n > SpectrumKernel.NMAX:
            warnings.warn(f"Becomes memory expensive when n > {SpectrumKernel.NMAX}")
            self._patterns = Parallel(n_jobs=config.n_jobs)(delayed(join)(pattern) for pattern in patterns)
        else:
            self._patterns = list(map(join, patterns))

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
        """Computes pairwise normalized kernel values with constraint
        that all sequences must have same length
        """
        # Order arrays by dimensionality
        min_X, max_X, min_len, _ = SpectrumKernel.order_arrays(X1, X2)
        seq_min_len = min(map(len, np.hstack([X1, X2])))
        seq_max_len = max(map(len, np.hstack([X1, X2])))
        assert seq_min_len == seq_max_len, "All sequences must have same length"
        if seq_min_len < self.n:
            return 0
        else:
            # Initialize counting dictionnaries
            counts_min = {idx: {perm: 0 for perm in self.patterns} for idx in range(len(min_X))}
            counts_max = {idx: {perm: 0 for perm in self.patterns} for idx in range(len(max_X))}
            # Iterate over sequences and count mers occurences
            for idx, (seq1, seq2) in tqdm(enumerate(zip(min_X, max_X)), disable=self.verbose):
                for i in range(seq_max_len - self.n):
                    subseq1 = self._get_tuple(seq1, i)
                    counts_min[idx] = self._count_pattern(subseq1, counts_min[idx])
                    subseq2 = self._get_tuple(seq2, i)
                    counts_max[idx] = self._count_pattern(subseq2, counts_max[idx])
            # Complete iteration over larger datasets
            for idx, seq in tqdm(enumerate(max_X[min_len:]), disable=self.verbose):
                for i in range(seq_max_len - self.n):
                    subseq = self._get_tuple(seq, i)
                    counts_max[idx] = self._count_pattern(subseq, counts_max[idx])
            # Compute normalized inner product between spectral features
            feats1 = np.array([np.fromiter(foo.values(), dtype=np.float32) for foo in counts_min.values()])
            norms1 = np.linalg.norm(feats1, axis=1).reshape(-1, 1)
            feats2 = np.array([np.fromiter(foo.values(), dtype=np.float32) for foo in counts_max.values()])
            norms2 = np.linalg.norm(feats2, axis=1).reshape(-1, 1)
            return np.inner(feats1 / norms1, feats2 / norms2)
