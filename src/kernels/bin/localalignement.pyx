cimport numpy as np
import numpy as np
np.import_array()

cpdef float _evaluate(str seq1,
                      str seq2,
                      np.ndarray[np.complex128_t, ndim=2] S,
                      dict char2idx,
                      float e,
                      float d,
                      float beta):
    cdef list sequences = [seq1, seq2]
    sequences.sort(key=len)
    cdef str min_seq = sequences[0]
    cdef str max_seq = sequences[1]
    cdef int min_len = len(min_seq) + 1
    cdef int max_len = len(max_seq) + 1
    cdef np.ndarray[np.complex128_t, ndim=2] M = np.zeros((min_len, max_len), dtype=np.complex128)
    cdef np.ndarray[np.complex128_t, ndim=2] X = np.zeros((min_len, max_len), dtype=np.complex128)
    cdef np.ndarray[np.complex128_t, ndim=2] Y = np.zeros((min_len, max_len), dtype=np.complex128)
    cdef np.ndarray[np.complex128_t, ndim=2] X2 = np.zeros((min_len, max_len), dtype=np.complex128)
    cdef np.ndarray[np.complex128_t, ndim=2] Y2 = np.zeros((min_len, max_len), dtype=np.complex128)
    for i in range(1, min_len):
        for j in range(1, max_len):
            M[i, j] = np.exp(beta * (S[char2idx[min_seq[i - 1]], char2idx[max_seq[j - 1]]]))\
                * (1 + X[i - 1, j - 1] + Y[i - 1, j - 1] + M[i - 1, j - 1])
            X[i, j] = np.exp(beta * d) * M[i - 1, j] + np.exp(beta * e) * X[i - 1, j]
            Y[i, j] = np.exp(beta * d) * (M[i, j - 1] + X[i, j - 1]) + \
                np.exp(beta * e) * Y[i, j - 1]
            X2[i, j] = M[i - 1, j] + X2[i - 1, j]
            Y2[i, j] = M[i, j - 1] + X2[i, j - 1] + Y2[i, j - 1]
    return np.complex128(1) + X2[-1, -1] + Y2[-1, -1] + M[-1, -1]
