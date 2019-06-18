"""
This file is meant to provide a kernel catalog, centralizing and simplifying
kernels import for computation sessions
"""

from .SpectrumKernel import SpectrumKernel
from .SubstringKernel import SubstringKernel
from .LocalAlignementKernel import LocalAlignementKernel
from .MismatchKernel import MismatchKernel
from .WeightedDegreeKernel import WDKernel
from .ShiftWeightedDegreeKernel import ShiftWDKernel
from .GaussianKernel import GaussianKernel


choices = {'spectrum': SpectrumKernel,
           'mismatch': MismatchKernel,
           'substring': SubstringKernel,
           'localalignement': LocalAlignementKernel,
           'weighted_degree': WDKernel,
           'shift_weighted_degree': ShiftWDKernel,
           'gaussian': GaussianKernel}


def choose(kernel_name):
    try:
        return choices[kernel_name]
    except KeyError:
        raise KeyError(f"Unkown kernel, please specify a key in {choices.keys()}")
