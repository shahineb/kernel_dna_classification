import os
import sys
import numpy as np

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from src.kernels.WeightedDegreeKernel import WDKernel
from utils.decorators import accepts


class ShiftWDKernel(WDKernel):

    @accepts(int, int, int)
    def __init__(self, n, shift, verbose=1):
        super(ShiftWDKernel, self).__init__(n, verbose)
        self._shift = shift

    @property
    def shift(self):
        return self._shift

    def _shift_weight(self, s):
        return 1 / (2 * (s + 1))

    def _fill_buffer(self, substr, buffer):
        """Fills buffer while parsing first chars

        Args:
            substr (str): new substring for buffer
            buffer (np.ndarray): (self.n, self.shift) record buffer of parsed mers for length
            in range(n) and with shift in range(shift)
        """
        for i in range(self.n):
            for s in range(self.shift + 1):
                if len(buffer[i, s]) == i + 1:
                    buffer[i, s] = buffer[i, s][1:] + substr[s]
                else:
                    buffer[i, s] = buffer[i, s] + substr[s]
        return buffer

    def _update_buffer(self, substr, buffer):
        """Updates buffer through parsing

        Args:
            substr (str): new substring for buffer
            buffer (np.ndarray): record buffer of parsed mers for length in {1, ..., n}
        """
        for i in range(self.n):
            for s in range(self.shift + 1):
                buffer[i, s] = buffer[i, s][1:] + substr[s]
        return buffer

    def _evaluate(self, seq1, seq2):
        """ We only apply forward shift as a forward shift for seq1
        wrt seq2 basically consist in a backward shift of seq2 wrt
        seq1

        Args:
            seq1 (str): dna sequence
            seq2 (str): dna sequence
        """
        seq_size = len(seq1)
        assert len(seq2) == seq_size, "Sequence must have identical length"
        assert seq_size > max(self.n, self.shift), "Sequence must be longer than max mer and shift size"
        # Initialize weights, buffers and cumulative sum
        kmer_weights = self._kmer_weight(np.arange(1, self.n + 1))
        shift_weights = self._shift_weight(np.arange(0, self.shift + 1)).reshape(-1, 1)
        buffer1 = np.empty((self.n, self.shift + 1), dtype='<U15')
        buffer2 = np.empty((self.n, self.shift + 1), dtype='<U15')
        cum_sum = 0
        # First start filling buffer and matching k-mers for k < n
        for i in range(self.n):
            buffer1 = self._fill_buffer(seq1[i: i + self.shift + 1], buffer1)
            buffer2 = self._fill_buffer(seq2[i: i + self.shift + 1], buffer2)
            matches1 = np.array([buffer1[:, 0] == row for row in buffer2.T]) * shift_weights
            matches2 = np.array([buffer2[:, 0] == row for row in buffer1.T]) * shift_weights
            cum_sum += np.sum(np.inner(matches1[:, :i + 1], kmer_weights[:i + 1]))
            cum_sum += np.sum(np.inner(matches2[:, :i + 1], kmer_weights[:i + 1]))
        # Continue sequences parsing up to sequence size - shift size
        for i in range(self.n, seq_size - self.shift):
            buffer1 = self._update_buffer(seq1[i: i + self.shift + 1], buffer1)
            buffer2 = self._update_buffer(seq2[i: i + self.shift + 1], buffer2)
            matches1 = shift_weights * np.array([buffer1[:, 0] == row for row in buffer2.T])
            matches2 = shift_weights * np.array([buffer2[:, 0] == row for row in buffer1.T])
            cum_sum += np.sum(np.inner(matches1, kmer_weights))
            cum_sum += np.sum(np.inner(matches2, kmer_weights))
        # Parse last chars as if no shift as involved
        buffer1 = buffer1[:, 0]
        buffer2 = buffer2[:, 0]
        for i in range(seq_size - self.shift, seq_size):
            buffer1 = super(ShiftWDKernel, self)._update_buffer(seq1[i], buffer1)
            buffer2 = super(ShiftWDKernel, self)._update_buffer(seq2[i], buffer2)
            cum_sum += np.inner(kmer_weights, buffer1 == buffer2)
        return cum_sum
