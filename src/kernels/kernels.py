from src.kernels.SpectrumKernel import SpectrumKernel
from src.kernels.SubstringKernel import SubstringKernel
from src.kernels.LocalAlignementKernel import LocalAlignementKernel
from src.kernels.GaussianKernel import GaussianKernel
from src.kernels.MismatchKernel import MismatchKernel
from src.kernels.WeightedDegreeKernel import WDKernel
from src.kernels.ShiftWeightedDegreeKernel import ShiftWDKernel

n = 5
charset = 'ACGT'

choices = {'spectrum': SpectrumKernel(n, charset),
           'substring': SubstringKernel,
           'localalignement': LocalAlignementKernel}

def choose(kernel_name):
    return choices[kernel_name]
