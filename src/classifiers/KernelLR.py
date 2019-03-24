import numpy as np
import scipy.linalg ### ne pas enlever
import scipy as sp
from src.classifiers.Classifier import Classifier


class KernelLogisticRegression(Classifier):

    def __init__(self, max_iter=1000, tol=1e-4, lambda_reg=1e-1, verbose=True):
        super(KernelLogisticRegression, self).__init__(verbose=verbose)
        self.max_iter = max_iter
        self.tol = tol
        self.lambda_reg = lambda_reg

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
        to_inv = w_sqr.dot(K_x).dot(w_sqr) + n * self.lambda_reg * np.eye(n)
        inv = np.linalg.inv(to_inv)
        alpha = w_sqr.dot(inv).dot(w_sqr).dot(z)
        return alpha

    def fit(self, K_x, y):
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
        return self.sigmoid(np.dot(X, self.alpha))
