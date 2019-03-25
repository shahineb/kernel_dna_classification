import os
import sys
import numpy as np

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from utils.decorators import accepts, fitted


class KernelPCA:
    """
    Implementation of the kernel PCA

    Args:
        n_components (int): number of components of the PCA
    """

    @accepts(int)
    def __init__(self, n_components=2):
        self._n_components = n_components
        self._fitted = False
        self._components = None
        self._gram_matrix = None
        self._explained_variance = None
        self._explained_variance_ratio = None
        self._singular_values = None

    @property
    def n_components(self):
        return self._n_components

    @property
    def components(self):
        return self._components

    @property
    @fitted
    def explained_variance(self):
        return self._explained_variance

    @property
    @fitted
    def explained_variance_ratio(self):
        return self._explained_variance_ratio

    @property
    @fitted
    def singular_values(self):
        return self._singular_values

    @property
    @fitted
    def gram_matrix(self):
        return self._gram_matrix

    def fit(self, gram_matrix):
        self._gram_matrix = gram_matrix
        # Centering
        n = np.shape(gram_matrix)[0]
        I, U = np.eye(n), (1 / n) * np.ones((n, n))
        centered_gram_matrix = (I - U).dot(gram_matrix).dot(I - U)

        # Fitting
        eigval, eigvec = np.linalg.eig(centered_gram_matrix)
        ordering = np.argsort(-eigval)
        eigval, eigvec = eigval[ordering], eigvec[:, ordering]
        self._components = eigvec[:, :self.n_components] / np.sqrt(eigval[:self.n_components])

        # Compute explained variance
        _, S, __ = np.linalg.svd(gram_matrix, full_matrices=False)
        _explained_variance = (S ** 2) / (n - 1)
        self._explained_variance = _explained_variance[:self.n_components]
        total_var = _explained_variance.sum()
        self._explained_variance_ratio = self._explained_variance / total_var
        _singular_values = S.copy()
        self._singular_values = _singular_values[:self.n_components]

        # Mark as fitted
        self._fitted = True

    @fitted
    def transform(self, M):
        # Projects M of size (p, n) on the principal components of gram_matrix (n, n)
        p, n = M.shape
        I, U, V = np.eye(n), (1 / n) * np.ones((p, n)), (1 / n) * np.ones((n, n)),
        centered_M = (M - U.dot(self.gram_matrix)).dot(I - V)
        return np.matmul(centered_M, self.components)
