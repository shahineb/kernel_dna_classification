"""
This file is meant to provide a kernel catalog, centralizing and simplifying
kernels import for computation sessions
"""

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
    return choices[kernel_name]
