from src.classifiers.KernelLR import KernelLogisticRegression


choices = {'kernel-lr': KernelLogisticRegression,
           'kernel-svm': 0}

def choose(name):
    return choices[name]