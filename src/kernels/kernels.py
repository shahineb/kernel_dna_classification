from src.kernels.SpectrumKernel import SpectrumKernel
from src.kernels.SubstringKernel import SubstringKernel
from src.kernels.LocalAlignementKernel import LocalAlignementKernel

n = 5
charset = 'ACGT'

choices = {'spectrum': SpectrumKernel(n, charset),
           'substring': SubstringKernel,
           'localalignement': LocalAlignementKernel
           }

def choose(kernel_name):
    return choices[kernel_name]