import os
import sys
import numpy as np
import cvxopt

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from src.classifiers.KernelSVM import KernelSVM


class Kernel2SVM(KernelSVM):
    """Implementation of Kernel squared hinge loss based on dual formulation

    Args:
        kernel (src.kernels.Kernel): kernel object
        lbda (float): regularization parameter
        support_vec_tol (float): tolerance for discarding non-supporting vectors
            if |alpha_i| < support_vec_tol * lbda then vector is discarded
        verbose (int): in {0, 1}
    """

    def __init__(self, kernel=None, lbda=1.0, support_vec_tol=1e-3, verbose=0):
        super(Kernel2SVM, self).__init__(kernel=kernel,
                                         lbda=lbda,
                                         support_vec_tol=support_vec_tol,
                                         verbose=verbose)

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
        Q = cvxopt.matrix(K + n * self._lbda * np.eye(n))
        p = -cvxopt.matrix(y)
        G = cvxopt.matrix(np.diag(-y))
        h = cvxopt.matrix(np.zeros(n))
        return Q, p, G, h
