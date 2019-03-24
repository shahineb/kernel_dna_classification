from SpectrumKernel import SpectrumKernel
from SubstringKernel import SubstringKernel
from LocalAlignementKernel import LocalAlignementKernel

choices = {'spectrum': SpectrumKernel,
           'substring': SubstringKernel,
           'localalignement': LocalAlignementKernel
           }

def choose(kernel_name):
    return choices[kernel_name]