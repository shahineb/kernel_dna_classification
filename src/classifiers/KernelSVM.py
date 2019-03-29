import os
import sys
import numpy as np
import cvxopt

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from src.classifiers.Classifier import Classifier
from utils.decorators import fitted


class KernelSVM(Classifier):
    """Implementation of Kernel SVM based on dual formulation

    Args:
        kernel (src.kernels.Kernel): kernel object
        lbda (float): regularization parameter
        support_vec_tol (float): tolerance for discarding non-supporting vectors
            if |alpha_i| < support_vec_tol * lbda then vector is discarded
        verbose (int): in {0, 1}
    """

    def __init__(self, kernel=None, lbda=1.0, support_vec_tol=1e-3, verbose=0):
        super(KernelSVM, self).__init__(kernel=kernel, verbose=verbose)
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

    def _setup_cvxopt_problem(self, K, y):
        """Initializes instances for QP solver

        Args:
            K (np.ndarray): gram matrix
            y (np.ndarray): label in {-1, 1}

        Returns:
            Q (cvxopt.matrix): quadratic form
            p (cvxopt.matrix): linear form
            G (cvxopt.matrix): left hand inequality constraints terms
            h (cvxopt.matrix): right hand inequality constrains terms
        """
        n = len(K)
        Q = cvxopt.matrix(K)
        p = -cvxopt.matrix(y)
        G = cvxopt.matrix(np.vstack([np.diag(-y), np.diag(y)]))
        h = cvxopt.matrix(np.hstack([np.zeros(n), 0.5 * np.ones(n) / self.lbda]))
        return Q, p, G, h

    def fit(self, X, y):
        # Kernel matrix and labels formatting
        self._Xtr = X
        if self.kernel:
            K = self.kernel(X, X).astype(np.double)
        else:
            K = X.astype(np.double)
        y = KernelSVM.format_binary_labels(y).astype(np.double)

        # Setup cvxopt QP args
        Q, p, G, h = self._setup_cvxopt_problem(K, y)
        cvxopt.solvers.options['show_progress'] = self.verbose

        # Solve problem
        sol = cvxopt.solvers.qp(Q, p, G, h)
        self._alpha = np.array(sol['x']).ravel()
        self._support_vectors = np.where(np.abs(self._alpha) > self.support_vec_tol * self.lbda)[0]
        self._fitted = True

        if self.verbose:
            print(f"Model fitted : f{len(self._support_vectors )} support vectors")

    def predict_prob(self, X):
        raise RuntimeError("No probability prediction for SVM")

    def predict(self, X):
        """Predicts label of samples X

        Args:
            X (np.ndarray)
        """
        if self.kernel:
            foo = self.kernel(self.Xtr[self.support_vectors], X)
        else:
            foo = X[self.support_vectors]
        y_pred = np.sign(self.alpha[self.support_vectors] @ foo)
        return y_pred.astype(int)
