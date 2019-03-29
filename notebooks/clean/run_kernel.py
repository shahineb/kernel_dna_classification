import os
import sys
import numpy as np

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from src.kernels.kernels import MismatchKernel
from utils.DataLoader import DataLoader

loader = DataLoader()

X0, _, y0, __ = loader.get_train_val(k=0, val_size=0.)
X1, _, y1, __ = loader.get_train_val(k=1, val_size=0.)
X2, _, y2, __ = loader.get_train_val(k=2, val_size=0.)
Xte0 = loader.get_test(k=0)
Xte1 = loader.get_test(k=1)
Xte2 = loader.get_test(k=2)


kernel = MismatchKernel(n=8, k=1, charset="ATCG", verbose=1)


gram_matrix = kernel(X0[:15], X0[:15])
np.savetxt(X=gram_matrix, fname="foo.csv")

gram_matrix = kernel(X0, X0)
np.savetxt(X=gram_matrix, fname="mismatch9_00.csv")

gram_matrix = kernel(X1, X1)
np.savetxt(X=gram_matrix, fname="mismatch9_11.csv")

gram_matrix = kernel(X2, X2)
np.savetxt(X=gram_matrix, fname="mismatch9_22.csv")

gram_matrix = kernel(X0, Xte0)
np.savetxt(X=gram_matrix, fname="mismatch9_test_00.csv")

gram_matrix = kernel(X1, Xte1)
np.savetxt(X=gram_matrix, fname="mismatch9_test_11.csv")

gram_matrix = kernel(X2, Xte2)
np.savetxt(X=gram_matrix, fname="mismatch9_test_22.csv")
