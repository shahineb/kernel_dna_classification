import os
import sys
import numpy as np
import cvxopt

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from src.classifiers.Classifier import Classifier
from utils.decorators import fitted


class Kernel2SVM(Classifier):
    """Implementation of Kernel SVM based on dual formulation

    Args:
        kernel (src.kernels.Kernel): kernel object
        C (float): strength parameter ( = 1 / (2 * lambda * n))
        support_vec_tol (float): tolerance for discarding non-supporting vectors
            if |alpha_i| < support_vec_tol * C then vector is discarded
        verbose (int): in {0, 1}
    """

    def __init__(self, kernel=None, lbda=1.0, support_vec_tol=1e-3, verbose=0):
        super(Kernel2SVM, self).__init__(kernel=kernel, verbose=verbose)
        self._lbda = lbda
        self._support_vec_tol = support_vec_tol
        self._alpha = None
        self._support_vectors = None

    @property
    def lbda(self):
        return self._lbda

    @property
    def support_vec_tol(self):
        return self._support_vec_tol

    @property
    @fitted
    def alpha(self):
        return self._alpha

    @property
    @fitted
    def support_vectors(self):
        return self._support_vectors

    def fit(self, X, y, precomputed=True):
        # Kernel matrix and labels formatting
        self._Xtr = X
        if self.kernel:
            K = self.kernel(X, X).astype(np.double)
        else:
            K = X.astype(np.double)
        y = Kernel2SVM.format_binary_labels(y).astype(np.double)

        # Setup cvxopt QP args
        n = len(K)
        Q = cvxopt.matrix(K + n*self._lbda*np.eye(n))
        p = -cvxopt.matrix(y)
        G = cvxopt.matrix(np.diag(-y))
        h = cvxopt.matrix(np.zeros(n))
        cvxopt.solvers.options['show_progress'] = self.verbose

        # Solve problem
        sol = cvxopt.solvers.qp(Q, p, G, h)
        self._alpha = np.array(sol['x']).ravel()
        self._support_vectors = np.where(np.abs(self._alpha) > self._support_vec_tol*self._lbda)[0]
        self._fitted = True

    def predict_prob(self, X):
        raise RuntimeError("No probability prediction for SVM")

    def predict(self, X, precomputed=False):
        if self.kernel:
            foo = self.kernel(self.Xtr[self.support_vectors], X)
        else:
            foo = X.astype(np.double)[self.support_vectors]
        y_pred = np.sign(self.alpha[self.support_vectors] @ foo)
        return y_pred
