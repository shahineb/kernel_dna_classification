import os
import sys
import numpy as np
from tqdm import tqdm

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from src.kernels.Kernel import Kernel
from utils.decorators import accepts


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
        seqs = [seq1, seq2]
        seqs.sort(key=len)
        min_seq, min_len = seqs[0], len(seqs[0]) + 1
        max_seq, max_len = seqs[1], len(seqs[1]) + 1

        M = np.zeros((min_len, max_len))
        X = np.zeros((min_len, max_len))
        Y = np.zeros((min_len, max_len))
        X2 = np.zeros((min_len, max_len))
        Y2 = np.zeros((min_len, max_len))
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

    def _gram_matrix(self, X1, X2):
        len_X1 = len(X1)
        len_X2 = len(X2)

        gram_matrix = np.zeros((len_X1, len_X2), dtype=np.float32)

        seqs_norms = {1: dict(), 2: dict()}
        for i in tqdm(range(len_X1)):
            seqs_norms[1][i] = self._evaluate(X1[i], X1[i])
        for i in tqdm(range(len_X2)):
            seqs_norms[2][i] = self._evaluate(X2[i], X2[i])

        for i in tqdm(range(len_X1)):
            for j in range(len_X2):
                if X1[i] == X2[j]:
                    gram_matrix[i, j] = 1
                else:
                    gram_matrix[i, j] = self._evaluate(X1[i], X2[j]) / (seqs_norms[1][i] * seqs_norms[2][i]) ** 0.5

        return gram_matrix
