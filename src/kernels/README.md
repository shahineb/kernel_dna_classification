# Kernels implementation

See [wiki on kernel manipulation](https://github.com/shahineb/kernel_challenge/wiki/Kernels-manipulation)

## Spectrum Kernel
(Leslie et al. 2002)

`SpectrumKernel` is initialized by :
  - `n` : mer length to be considered
  - `charset` : string with all chars to be considered (e.g "ATCG")

It's based on the number of occurences of each n-uple made out of the provided set of chars. The proposed implementation presents a preindexed version of this kernel, making it hence very computationally efficient. However, we recommend not to use a tuple lenght greater than 7.

## Mismatch Kernel
(Leslie et al. 2003)

`MismatchKernel` is initialized by :
  - `n` : mer length to be considered
  - `charset` : string with all chars to be considered (e.g "ATCG")
  - `k` : maxmimum number of mismatch allowed

It is very similar to the Spectrum Kernel expect for the fact that for a given sequence, we count occurences of each n-uples along with their k-neighbors. For example, with n=3 and k=1, if we parse sequence `'AAA'`, it would also count for `{'AAC', 'AAG', 'AAT', 'ACA', 'AGA', 'ATA', 'CAA', 'GAA', 'TAA'}`.

## Weighted Degree Kernel
(Ratsch and Sonnenburg, 2004)

`WDKernel` is initialized by:
  - `n` : maximal mer length to be considered

Compares co-occurences of k-mers at corresponding position in sequences. Implementation is performed in a linear time by parsing characters successively and keeping track of last k-mers parsed in a buffer.

## Substring kernel
(Lodhi et al. 2002)

`SubstringKernel` is initialized by :
  - `n` : strings length to be considered
  - `decay_rate` : decay parameter in ]0,1[

It's based on substring occurences, with length and spacing penalization.

## Local Alignement Kernel
(Vert et al. 2004)

`LocalAlignementKernel` is initialized by :
  - `S` : substitution matrix, i.e. cost/reward of substituting a nucleotide by another as numpy array
  - `char2idx` : mapping from nucleotide to matching index in `S`
  - `e` : affine gap penalty slope
  - `d` : affine gap penalty intercept
  - `beta` : local alignement parameter

Based on the Smith Waterman score, this kernel tries to quantify alignement performance between 2 DNA sequences.

Accordingly with the parameters used for experiments in the paper, we set the parameters `e` = 11, `d` = 1 and use the [BLOSUM62 similarity matrix](https://fr.wikipedia.org/wiki/Matrice_de_similarit%C3%A9#Exemple) given by :


|   | A | T  | C  | G  |
|---|---|----|----|----|
| A | 4 | 0  | 0  | 0  |
| T | 0 | 5  | -1 | -2 |
| C | 0 | -1 | 9  | -3 |
| G | 0 | -2 | -3 | 6  |

Plus, we perform spectral translation to cope with diagonal dominance issue. We hence choose `beta`=0.5 given ROC scores observed in the paper.

Please note this kernel is computationally slow.
