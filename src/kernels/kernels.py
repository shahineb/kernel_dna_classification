"""
This file is meant to provide a kernel catalog, centralizing and simplifying
kernels import for computation sessions
"""
import os
import sys

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)


from src.kernels.SpectrumKernel import SpectrumKernel
from src.kernels.SubstringKernel import SubstringKernel
from src.kernels.LocalAlignementKernel import LocalAlignementKernel
from src.kernels.MismatchKernel import MismatchKernel
from src.kernels.WeightedDegreeKernel import WDKernel
from src.kernels.ShiftWeightedDegreeKernel import ShiftWDKernel
from src.kernels.GaussianKernel import GaussianKernel


choices = {'spectrum': SpectrumKernel,
           'mismatch': MismatchKernel,
           'substring': SubstringKernel,
           'localalignement': LocalAlignementKernel,
           'weighted_degree': WDKernel,
           'shift_weighted_degree': ShiftWDKernel}


def choose(kernel_name):
    try:
        return choices[kernel_name]
    except KeyError:
        raise KeyError(f"Unkown kernel, please specify a key in {choices.keys()}")
