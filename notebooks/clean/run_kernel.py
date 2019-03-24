import os
import sys
import numpy as np

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from src.kernels.kernels import LocalAlignementKernel
from utils.DataLoader import DataLoader

loader = DataLoader()

X0, _, y0, __ = loader.get_train_val(k=0, val_size=0.)
X1, _, y1, __ = loader.get_train_val(k=1, val_size=0.)
X2, _, y2, __ = loader.get_train_val(k=2, val_size=0.)
Xte0 = loader.get_test(k=0)
Xte1 = loader.get_test(k=1)
Xte2 = loader.get_test(k=2)


S = np.array([[3, -1, -1, -1],
              [-1, 8, -4, -2],
              [-1, -4, 5, -3],
              [-1, -2, -3, 4]], dtype=np.float64)
char2idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
beta = 0.01
e = 11.
d = 1.

kernel = LocalAlignementKernel(S=S,
                               char2idx=char2idx,
                               e=e,
                               d=d,
                               beta=beta)


gram_matrix = kernel(X0, X0)
np.savetxt(X=gram_matrix, fname="localalignement_00.csv")

gram_matrix = kernel(X1, X1)
np.savetxt(X=gram_matrix, fname="localalignement_11.csv")

gram_matrix = kernel(X2, X2)
np.savetxt(X=gram_matrix, fname="localalignement_22.csv")

gram_matrix = kernel(X0, Xte0)
np.savetxt(X=gram_matrix, fname="localalignement_test_00.csv")

gram_matrix = kernel(X1, Xte1)
np.savetxt(X=gram_matrix, fname="localalignement_test_11.csv")

gram_matrix = kernel(X2, Xte2)
np.savetxt(X=gram_matrix, fname="localalignement_test_22.csv")
