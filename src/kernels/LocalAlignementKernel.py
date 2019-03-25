import os
import sys
import numpy as np
import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()})

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from src.kernels.Kernel import Kernel
from utils.decorators import accepts


class LocalAlignementKernel(Kernel):
    """Implementation of Vert et al., 2004
    for affine gap penalty function.
    We use Spectral Translation to overcome non-positive-definiteness
    when tackling diagonal dominance issue with log
    """

    BLOSUM62 = np.array([[4., 0., 0., 0.],
                         [0., 5., -1., -2.],
                         [0., -1., 9., -3.],
                         [0., -2., -3., 6.]], dtype=np.float64)
    CHAR2IDX = {'A': 0, 'T': 1, 'C': 2, 'G': 3}

    @accepts(np.ndarray, dict, float, float, float, int)
    def __init__(self,
                 S=BLOSUM62,
                 char2idx=CHAR2IDX,
                 e=11., d=1., beta=0.5, verbose=1):
        """
        Args:
            S (np.ndarray): substitution matrix
            char2idx (dict) : mapping from char to indexes in substitution matrix
            e (float): affine gap penalty slope
            d (float): affine gap penalty intercept
            beta (float): local alignement parameter
        """
        super(LocalAlignementKernel, self).__init__(verbose)
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
        seqs = [seq1, seq2]
        seqs.sort(key=len)
        min_seq, min_len = seqs[0], len(seqs[0]) + 1
        max_seq, max_len = seqs[1], len(seqs[1]) + 1

        M = np.zeros((min_len, max_len), dtype=np.float128)
        X = np.zeros((min_len, max_len), dtype=np.float128)
        Y = np.zeros((min_len, max_len), dtype=np.float128)
        X2 = np.zeros((min_len, max_len), dtype=np.float128)
        Y2 = np.zeros((min_len, max_len), dtype=np.float128)
        for i in range(1, min_len):
            for j in range(1, max_len):
                M[i, j] = np.exp(self.beta * (self.S[self.char2idx[min_seq[i - 1]], self.char2idx[max_seq[j - 1]]]))\
                    * (1 + X[i - 1, j - 1] + Y[i - 1, j - 1] + M[i - 1, j - 1])
                X[i, j] = np.exp(self.beta * self.d) * M[i - 1, j] + np.exp(self.beta * self.e) * X[i - 1, j]
                Y[i, j] = np.exp(self.beta * self.d) * (M[i, j - 1] + X[i, j - 1]) + \
                    np.exp(self.beta * self.e) * Y[i, j - 1]
                X2[i, j] = M[i - 1, j] + X2[i - 1, j]
                Y2[i, j] = M[i, j - 1] + X2[i, j - 1] + Y2[i, j - 1]
        return 1 + X2[-1, -1] + Y2[-1, -1] + M[-1, -1]

    def _pairwise(self, X1, X2):
        pairwise_matrix = super(LocalAlignementKernel, self)._pairwise(X1, X2)
        try:
            buffer = np.log(pairwise_matrix) / self.beta
            min_eigen_value = np.min(np.linalg.eigvals(buffer))
            pairwise_matrix = buffer - (min(0, min_eigen_value) - 1) * np.eye(len(buffer))
        except np.linalg.LinAlgError:
            pass
        return pairwise_matrix
