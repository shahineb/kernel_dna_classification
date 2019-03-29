"""
Couple of utilities for kernel computation such as dominant diagonal handling
functions
"""

import numpy as np


def spectral_translation(gram_matrix, set_off=np.finfo(np.float16).eps):
    """Performs spectral translation of a non-p.d gram matrix
    """
    min_eigen_val = min(0, np.min(np.linalg.eigvalsh(gram_matrix)) - set_off)
    return gram_matrix - min_eigen_val * np.eye(len(gram_matrix))


def empirical_kernel_map(gram_matrix):
    """ Implementation proposed by Scholkopf et al. (2002)
    """
    return gram_matrix @ gram_matrix.T
