import os
import sys
import numpy as np

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from src.classifiers.Classifier import Classifier
from utils.decorators import fitted


class MultipleKernelLearner(Classifier):
    """Implementation of Multiple Kernel Learning with reduced gradient method.
    (http://www.jmlr.org/papers/volume9/rakotomamonjy08a/rakotomamonjy08a.pdf)

    Args:
        clf (KernelSVM): base SVM classifier used
        M (int): number of kernels
        eta_init (np.array): initialization of weight parameter (same size as gram_matrices_list)
        lr (float): learning rate for weight update
        eta_tol (float): tolerance for convergence of weight updates
        max_iter (int): maximal number of iterations for weight updates
        verbose (int): in {0, 1}
    """

    def __init__(self, clf, M, eta_init=None, lr=1e2, eta_tol=5e-2, max_iter=100, verbose=0):
        super(MultipleKernelLearner, self).__init__(kernel=clf.kernel, verbose=verbose)
        self._clf = clf
        self._M = M
        if eta_init is None:
            self._eta = (1 / self._M) * np.ones(self._M)
        else:
            self._eta = eta_init
        self._lr = lr
        self._eta_tol = eta_tol
        self._max_iter = max_iter

    @property
    def M(self):
        return self._M

    @property
    def clf(self):
        return self._clf

    @property
    def lr(self):
        return self._lr

    @property
    def eta_tol(self):
        return self._eta_tol

    @property
    def max_iter(self):
        return self._max_iter

    @property
    @fitted
    def eta(self):
        return self._eta

    def fit(self, gram_matrices_list, y):
        """
        Args:
            gram_matrices_list (list[np.ndarray]): list of gram matrices to fit
            y (np.ndarray): label
        """
        assert len(gram_matrices_list) == self.M, f"Number of gram matrices must be {self.M}"
        lbda = self.clf.lbda
        c, convergence = 0, False
        while (c < self.max_iter & ~convergence):
            gram_matrix_mkl = sum(self._eta[i] * gram_matrices_list[i] for i in range(self.M))
            self.clf.fit(gram_matrix_mkl, y)
            gamma = 2 * lbda * y * self.clf._alpha
            grad_eta = np.array([-gamma.T @ gram_matrices_list[i] @ gamma for i in range(self.M)])
            eta_arg_max = np.argmax(self._eta)
            grad_eta = (grad_eta - grad_eta[eta_arg_max]) * (np.array(self._eta) > 0)
            grad_eta[eta_arg_max] = -np.sum(grad_eta)
            old_eta = self._eta.copy()
            self._eta += self.lr * grad_eta
            self._eta = np.clip(self._eta, 0, 1)
            convergence = (np.linalg.norm(self._eta - old_eta) > self.eta_tol)
            if convergence:
                self._fitted = True

    def predict_prob(self, gram_matrices_list):
        raise RuntimeError("No probability prediction for SVM")

    def predict(self, gram_matrices_list):
        """Predicts label of samples X

        Args:
            gram_matrices_list (list[np.ndarray]): list of gram matrices
        """
        gram_matrix_mkl = sum(self.eta[i] * gram_matrices_list[i] for i in range(self.M))
        return self.clf.predict(gram_matrix_mkl)
