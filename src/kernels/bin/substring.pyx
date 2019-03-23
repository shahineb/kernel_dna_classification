cimport numpy as np
import numpy as np
np.import_array()


cpdef float _Kprime(str seq1, str seq2, int depth, int n, float decay_rate):
    cdef int len_1 = len(seq1)
    cdef int len_2 = len(seq2)
    cdef float part_sum = 0.
    cdef float result
    if depth == 0:
        return 1.
    elif min(len_1, len_2) < depth:
        return 0.
    else:
        for j in range(1, len_2):
            if seq2[j] == seq1[-1]:
                part_sum += (decay_rate ** (len_2 - (j + 1) + 2)) * _Kprime(seq1[:-1],
                                                                            seq2[:j],
                                                                            depth - 1,
                                                                            n,
                                                                            decay_rate)
        result = part_sum + decay_rate * _Kprime(seq1[:-1],
                                                 seq2,
                                                 depth,
                                                 n,
                                                 decay_rate)
        return result


cpdef float _evaluate(str seq1, str seq2, int n, float decay_rate):
    cdef int len_1 = len(seq1)
    cdef int len_2 = len(seq2)
    min_len = min(len_1, len_2)
    cdef float part_sum = 0.
    cdef float result
    if min_len < n:
        return 0.
    else:
        for j in range(1, len_2):
            if seq2[j] == seq1[-1]:
                part_sum += _Kprime(seq1[:-1], seq2[:j], n - 1, n, decay_rate)
        result = _evaluate(seq1[:-1], seq2, n, decay_rate) + decay_rate ** 2 * part_sum
        return result

cpdef np.ndarray[np.float64_t, ndim=2] _empty_gram(int n, int p):
    cdef np.ndarray[np.float64_t, ndim=2] gram_matrix = np.zeros((n, p), dtype=np.float64)
    return gram_matrix


cpdef np.ndarray[np.float64_t, ndim=2] _gram_matrix(np.ndarray X1, np.ndarray X2, int n, float decay_rate):
    cdef list X = [X1, X2]
    X.sort(key=len)
    cdef np.ndarray min_X = X[0]
    cdef np.ndarray max_X = X[1]
    cdef int min_len = len(X[0])
    cdef int max_len =  len(X[1])
    cdef np.ndarray[np.float64_t, ndim=2] gram_matrix = _empty_gram(min_len, max_len)
    cdef float buffer
    seqs_norms = {0: dict(), 1: dict()}


    for i in range(min_len):
        buffer = _evaluate(min_X[i], min_X[i], n, decay_rate)
        seqs_norms[0][i] = buffer
        if min_X[i] == max_X[i]:
            seqs_norms[1][i] = buffer
        else:
            seqs_norms[1][i] = _evaluate(max_X[i], max_X[i], n, decay_rate)
    for i in range(min_len, max_len):
        seqs_norms[1][i] = _evaluate(max_X[i], max_X[i], n, decay_rate)
    for i in range(min_len):
        for j in range(max_len):
            if min_X[i] == max_X[j]:
                gram_matrix[i, j] = 1
            else:
                gram_matrix[i, j] = _evaluate(min_X[i], max_X[j], n, decay_rate) / (seqs_norms[0][i] * seqs_norms[1][j]) ** 0.5
    return gram_matrix
