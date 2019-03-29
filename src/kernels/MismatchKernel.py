import os
import sys
from joblib import Parallel, delayed


base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from src.kernels.SpectrumKernel import SpectrumKernel
from utils.decorators import accepts
import config


class MismatchKernel(SpectrumKernel):
    """Implementation of Leslie et al., 2003
    with strings preindexation
    """

    @staticmethod
    def substitution(word, char, pos):
        return word[:pos] + char + word[pos + 1:]

    @staticmethod
    def _generate_neighbor(word, alphabet, k):
        """Generates all possible mismatching neighbors with levenshtein
        distance lower than k

        Args:
            word (str): seed word
            alphabet (str): charset to use for substitution
            k (int): maximum levenshtein distance authorized in misspelling
        """
        neighbors = []
        for i, char in enumerate(word):
            for l in alphabet:
                neighbors += [MismatchKernel.substitution(word, l, i)]
        if k > 1:
            neighbors += MismatchKernel._generate_neighbor(word, alphabet, k - 1)
        return neighbors

    @accepts(int, int, str, int)
    def __init__(self, n, k, charset, verbose=0):
        """
        Args:
            n (int): n-mers size to consider
            k (int): number of mismatch allowed
            charset (str): charset for preindexation (typically "ATCG")
            verbose (int): {0, 1}
        """
        super(MismatchKernel, self).__init__(n=n,
                                             charset=charset,
                                             verbose=verbose)
        self._k = k
        if self._n > self.NMAX:
            helper = lambda pattern: (pattern, MismatchKernel._generate_neighbor(pattern, charset, k))
            items = Parallel(n_jobs=config.n_jobs)(delayed(helper)(pattern) for pattern in self._patterns)
            self._neighbors = {pattern: neighbors for (pattern, neighbors) in items}
        else:
            self._neighbors = {pattern: set(MismatchKernel._generate_neighbor(pattern, charset, k))
                               for pattern in self._patterns}

    @property
    def k(self):
        return self._k

    @property
    def neighbors(self):
        return self._neighbors

    def _count_pattern(self, pattern, count_dict):
        for neighbor in self.neighbors[pattern]:
            count_dict[neighbor] += 1
        return count_dict
