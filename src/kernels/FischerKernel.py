import os
import sys
import warnings
from six import string_types
from itertools import product
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from src.kernels.Kernel import Kernel
from utils.decorators import accepts
import config


class FischerKernel(Kernel):

    NMAX = 8

    def __init__(self, hmm, verbose=0):
        """
        Args:
            n (int): n-mers size to consider
            charset (str): charset for preindexation (typically "ATCG")
            verbose (int): {0, 1}
        """
        super(FischerKernel, self).__init__(hmm, verbose)
        self.hmm = hmm

    @staticmethod
    def transform_letter_in_one_hot_vector(letter):
        if letter == 'A':
            return [1, 0, 0, 0]
        elif letter == 'C':
            return [0, 1, 0, 0]
        elif letter == 'G':
            return [0, 0, 1, 0]
        elif letter == 'T':
            return [0, 0, 0, 1]

    @staticmethod
    def transform_seq_into_spare_hot_vector(sequence):
        vector = [FischerKernel.transform_letter_in_one_hot_vector(letter) for letter in sequence]
        return np.array(vector)

    def get_fischer_vector(self, X):
        '''
        X: (n, p) np.array data matrix

        Returns:
        q: most probable latent variables
        '''

        # compute forward and backward messages
        log_alpha = self.hmm.build_log_alpha(X)
        log_beta = self.hmm.build_log_beta(X)

        # compute probabilities for latent variables (E-step)
        prob_unary = self.hmm.unary_prob(log_alpha, log_beta)
        return prob_unary - self.hmm.pi_0

    def _evaluate(self, seq1, seq2):
        """
        Args:
            seq1 (str): dna sequence
            seq2 (str): dna sequence
        """
        raise NotImplementedError

    def _pairwise(self, X1, X2):
        """Computes pairwise normalized kernel values with constraint
        that all sequences must have same length
        """
        data = FischerKernel.transform_seq_into_spare_hot_vector(X1)
        self.hmm.fit(data)
        return self.get_fischer_vector(X1).T.dot(self.get_fischer_vector(X2))
