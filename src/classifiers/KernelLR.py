import os
import sys
import numpy as np
import scipy.linalg ### ne pas enlever
import scipy as sp

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from src.classifiers.Classifier import Classifier


class KernelLogisticRegression(Classifier):
    """An implementation of Kernel logistic regression

    Args:
        max_iter (int): maximum number of iterations
        tol (float): tolerance threshold for convergence criterion
        lbda (float): regularization parameter
        verbose (int): in {0, 1}
    """

    def __init__(self, kernel=None, max_iter=1000, tol=1e-4, lbda=1e-1, verbose=True):
        super(KernelLogisticRegression, self).__init__(kernel=kernel, verbose=verbose)
        self._max_iter = max_iter
        self._tol = tol
        self._lbda = lbda

    @property
    def max_iter(self):
        return self._max_iter

    @property
    def tol(self):
        return self._tol

    @property
    def lbda(self):
        return self._lbda

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def loss(h, y):
        return 0.5 * (- (1 + y) * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def _update_irls_m(self, K_x):
        return K_x.dot(self.alpha)

    def _update_irls_w(self, m):
        return np.nan_to_num(np.diag(self.sigmoid(m)[:, 0]) * np.diag(self.sigmoid(-m)[:, 0]))

    def _update_irls_z(self, m, y):
        return m + y / self.sigmoid(-y * m)

    def _update_irls_alpha(self, K_x, w, z):
        w_sqr = sp.linalg.sqrtm(w)
        n = K_x.shape[0]
        to_inv = w_sqr.dot(K_x).dot(w_sqr) + n * self.lbda * np.eye(n)
        inv = np.linalg.inv(to_inv)
        alpha = w_sqr.dot(inv).dot(w_sqr).dot(z)
        return alpha

    def fit(self, X, y):
        self._Xtr = X
        if self.kernel:
            K_x = self.kernel(X, X)
        else:
            K_x = X.copy()
        y = np.expand_dims(y, axis=-1)
        # weights initialization
        self.alpha = np.random.randn(K_x.shape[1], 1)

        for i in range(self.max_iter):
            old_alpha = self.alpha
            m = self._update_irls_m(K_x)
            W = self._update_irls_w(m)
            z = self._update_irls_z(m, y)
            self.alpha = self._update_irls_alpha(K_x, W, z)

            if self._verbose:
                print(f'Alpha loss: {np.linalg.norm(self.alpha - old_alpha)} \t')
            if np.linalg.norm(self.alpha - old_alpha) < self.tol:
                print('Convergence!')
                break

    def predict_prob(self, X):
        """Predicts proba of samples X

        Args:
            X (np.ndarray)
        """
        if self.kernel:
            foo = self.kernel(self.Xtr, X)
        else:
            foo = X
        return self.sigmoid(np.dot(foo, self.alpha))
