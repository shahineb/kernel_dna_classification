"""
This file is meant to provide a classifier catalog, centralizing and simplifying
classfiers import for computation sessions
"""
import os
import sys

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from src.classifiers.KernelLR import KernelLogisticRegression
from src.classifiers.KernelSVM import KernelSVM
from src.classifiers.Kernel2SVM import Kernel2SVM

choices = {'kernel-lr': KernelLogisticRegression,
           'kernel-svm': KernelSVM,
           'kernel-2svm': Kernel2SVM}


def choose(clf_name):
    try:
        return choices[clf_name]
    except KeyError:
        raise KeyError(f"Unkown classifier, please specify a key in {choices.keys()}")
