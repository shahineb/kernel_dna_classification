import os
import sys
import numpy as np

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from src.kernels.Kernel import StringKernel
from utils.decorators import accepts


class WDKernel(StringKernel):
    """Implementation of (Ratsch and Sonnenburg, 2004)
    """

    @accepts(int, int)
    def __init__(self, n, verbose=1):
        super(WDKernel, self).__init__(n, verbose)

    def _kmer_weight(self, k):
        """Computes mer weight given Ratsch and Sonnenburg, 2004

        Args:
            k (int): mer's length
        """
        return 2 * (self.n - k + 1) / (self.n * (self.n + 1))

    def _fill_buffer(self, char, buffer):
        """Fills buffer while parsing first chars

        Args:
            char (str): new character for buffer
            buffer (np.ndarray): (self.n, )record buffer of parsed mers for length in {1, ..., n}
        """
        for i in range(self.n):
            if len(buffer[i]) == i + 1:
                buffer[i] = buffer[i][1:] + char
            else:
                buffer[i] = buffer[i] + char
        return buffer

    def _update_buffer(self, char, buffer):
        """Updates buffer through parsing

        Args:
            char (str): new character for buffer
            buffer (np.ndarray): record buffer of parsed mers for length in {1, ..., n}
        """
        for i in range(self.n):
            buffer[i] = buffer[i][1:] + char
        return buffer

    def _evaluate(self, seq1, seq2):
        seq_size = len(seq1)
        assert len(seq2) == seq_size, "Sequence must have identical length"
        assert seq_size > self.n, "Sequence must be longer than max mer size"
        # Initialize weights, buffers and cumulative sum
        weights = self._kmer_weight(np.arange(1, self.n + 1))
        buffer1 = np.empty(self.n, dtype='<U15')
        buffer2 = np.empty(self.n, dtype='<U15')
        cum_sum = 0

        # First start filling buffer and matching k-mers for k < n
        for i in range(self.n):
            buffer1 = self._fill_buffer(seq1[i], buffer1)
            buffer2 = self._fill_buffer(seq2[i], buffer2)
            cum_sum += np.inner(weights[:i + 1], buffer1[:i + 1] == buffer2[:i + 1])
        # Continue sequences parsing
        for i in range(self.n, seq_size):
            buffer1 = self._update_buffer(seq1[i], buffer1)
            buffer2 = self._update_buffer(seq2[i], buffer2)
            cum_sum += np.inner(weights, buffer1 == buffer2)
        return cum_sum
