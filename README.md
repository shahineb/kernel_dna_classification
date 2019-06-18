# Kernel Methods for learning : DNA classification

### Challenge description

_Transcription factors (TFs) are regulatory proteins that bind specific sequence motifs in the genome to activate or repress transcription of target genes. Genome-wide protein-DNA binding maps can be profiled using some experimental techniques and thus all genomics can be classified into two classes for a TF of interest: bound or unbound. In this challenge, we will work with three datasets corresponding to three different TFs._

[Link to the challenge](https://www.kaggle.com/c/kernel-methods-for-machine-learning-2018-2019)

### Repository description

Repo is organized as follows :

```
├── data
├── demo
├── docs
│   ├── svg
│   └── tex
├── src
│   ├── classifiers
│   ├── decomposition
│   ├── evaluation
│   └── kernels
└── utils
```

- `data`: contains provided datasets (`Xtr[012].csv`, `Ytr[012].csv` and `Xte[012].csv`) as well as precomputed kernels
- `demo`: demonstration notebooks
- `doc`: images used in repository (`/svg`) and [project report](https://github.com/shahineb/kernel_challenge/blob/master/docs/tex/report.pdf) (`/tex`)
- `src`:
  - `classifiers`: kernel-based classifiers such as Kernel Logistic Regression or Kernel SVM
  - `decomposition`: kernel-based matrix decomposition algorithm (so far only Kernel PCA)
  - `evaluation`: evaluation metrics and model selection scripts
  - `kernels`: various kernels implementation for DNA sequence comparison, see [wiki](https://github.com/shahineb/kernel_challenge/wiki/Kernels-manipulation)

### Results

Submission file is stored under `submission.csv` and can be reproduced by running `python train.py`

- Team name : Kernelito

|         | Score (acc) | Ranking |
|---------|-------------|---------|
| Public  | 0.72066     | 10      |
| Private | 0.69733     | 15      |
