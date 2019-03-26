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
        classifier (KernelSVM): base SVM classifier used
        M (int): number of kernels
        eta_init (np.array): initialization of weight parameter (same size as gram_matrices_list)
        lr (float): learning rate for weight update
        eta_tol (float): tolerance for convergence of weight updates
        maxiter (int): maximal number of iterations for weight updates
        verbose (int): in {0, 1}
    """
    
    def __init__(self, classifier, M, eta_init=None, lr=1e2, eta_tol=5e-2, maxiter=100, verbose=0):
        super(MultipleKernelLearner, self).__init__(kernel=classifier._kernel, verbose=verbose)
        self._classifier = classifier
        self._M = M
        if eta_init is None:
            self._eta = (1/self._M) * np.ones(self._M)
        else:
            self._eta = eta_init
        self._lr = lr
        self._eta_tol = eta_tol
        self._maxiter = maxiter
        self._converged = False


    @property
    def eta(self):
        return self._eta


    def fit(self, gram_matrices_list, y):
        lbda = 1 / (2*self._classifier._C*y.shape[0])
        c, convergence = 0, False
        while (c<self._maxiter & ~convergence):
            gram_matrix_mkl = sum(self._eta[i]*gram_matrices_list[i] for i in range(self._M))
            self._classifier.fit(gram_train_mkl, y)
            gamma = 2 * lbda * y * self._classifier._alpha
            grad_eta = np.array([ - gamma.T @ gram_matrices_list[i] @ gamma for i in range(self._M) ])
            eta_arg_max = np.argmax(self._eta)
            grad_eta = (grad_eta - grad_eta[eta_arg_max]) * (np.array(self._eta)>0)
            grad_eta[eta_arg_max] = -np.sum(grad_eta)
            old_eta = self._eta.copy()
            self._eta += self._lr*grad_eta
            self._eta = np.clip(self._eta, 0 , 1)
            convergence = (np.linalg.norm(self._eta - old_eta) > self._eta_tol)
            if convergence:
                self._converged = True
            
    def predict_prob(self, gram_matrices_list):
        raise RuntimeError("No probability prediction for SVM")

    def predict(self, gram_matrices_list):
        gram_matrix_mkl = sum(self._eta[i]*gram_matrices_list[i] for i in range(self._M))
        return self._classifier.predict(gram_matrix_mkl, precomputed=True)