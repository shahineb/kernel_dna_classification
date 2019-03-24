import os
import sys
import numpy as np

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

# from src.kernels.SpectrumKernel import SpectrumKernel
from src.kernels.SubstringKernel import SubstringKernel
# from src.kernels.LocalAlignementKernel import LocalAlignementKernel
from utils.DataLoader import DataLoader

loader = DataLoader()

X = loader.load("Xtr0.csv", as_array=True)
kernel = SubstringKernel(n=2, decay_rate=0.3)


gram_matrix = kernel(X, X)
np.savetxt(X=gram_matrix, fname="substring_n2_decay_rate0.3_Xtr0.csv")
