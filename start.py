import os
import sys
import numpy as np
import argparse

cur_dir = os.getcwd()
base_dir = os.path.dirname(os.path.dirname(cur_dir))
sys.path.append(base_dir)

from src.classifiers.classifiers import Kernel2SVM
from utils.DataLoader import DataLoader

def get_labels(loader):
    _, _, ytr0, _ = loader.get_train_val(0, 0.)
    _, _, ytr1, _ = loader.get_train_val(1, 0.)
    _, _, ytr2, _ = loader.get_train_val(2, 0.)
    return ytr0, ytr1, ytr2

def load_precomputed_kernels(loader):
    # Load precomputed Kernels and first Dataset Kernel preparation
    X0_9 = np.loadtxt(os.path.join(loader.data_dir, "precomputed/mismatch_n9_k2/mismatch_n9_k2_00.csv"))
    X0_8 = np.loadtxt(os.path.join(loader.data_dir, "precomputed/mismatch_n8_k1/mismatch_n8_k1_00.csv"))
    Xte0_9 = np.loadtxt(os.path.join(loader.data_dir, "precomputed/mismatch_n9_k2/mismatch_n9_k2_test_00.csv"))
    Xte0_8 = np.loadtxt(os.path.join(loader.data_dir, "precomputed/mismatch_n8_k1/mismatch_n8_k1_test_00.csv"))
    X0 = X0_9 + X0_8
    Xte0 = Xte0_9 + Xte0_8
    X1 = np.loadtxt(os.path.join(loader.data_dir, "precomputed/mismatch_n9_k1/mismatch_n9_k1_11.csv"))
    Xte1 = np.loadtxt(os.path.join(loader.data_dir, "precomputed/mismatch_n9_k1/mismatch_n9_k1_test_11.csv"))
    X2_9 = np.loadtxt(os.path.join(loader.data_dir, "precomputed/mismatch_n9_k2/mismatch_n9_k2_22.csv"))
    X2_8 = np.loadtxt(os.path.join(loader.data_dir, "precomputed/mismatch_n8_k1/mismatch_n8_k1_22.csv"))
    Xte2_9 = np.loadtxt(os.path.join(loader.data_dir, "precomputed/mismatch_n9_k2/mismatch_n9_k2_test_22.csv"))
    Xte2_8 = np.loadtxt(os.path.join(loader.data_dir, "precomputed/mismatch_n8_k1/mismatch_n8_k1_test_22.csv"))
    X2 = X2_9 + X2_8
    Xte2 = Xte2_9 + Xte2_8
    return X0, Xte0, X1, Xte1, X2, Xte2

def compute_kernels(loader):
    from src.kernels.MismatchKernel import MismatchKernel as Mismatch
    mism8 = Mismatch(n=8, k=1, charset="ATCG", verbose=1)
    mism9 = Mismatch(n=9, k=2, charset="ATCG", verbose=1)

    X0, _, y0, __ = loader.get_train_val(k=0, val_size=0.)
    X1, _, y1, __ = loader.get_train_val(k=1, val_size=0.)
    X2, _, y2, __ = loader.get_train_val(k=2, val_size=0.)
    Xte0 = loader.get_test(k=0)
    Xte1 = loader.get_test(k=1)
    Xte2 = loader.get_test(k=2)
    X0 = mism8(X0, X0) + mism9(X0, X0)
    X1 = mism9(X1, X1)
    X2 = mism8(X2, X2) + mism9(X2, X2)
    return X0, Xte0, X1, Xte1, X2, Xte2


def get_kernels(args, loader):
    if args.precompute:
        return compute_kernels(loader)
    else:
        return load_precomputed_kernels(loader)


def predict(X, y, lbda, Xte):
    # Prediction on the first dataset
    svm = Kernel2SVM(kernel=None, lbda=lbda, support_vec_tol=0.01)
    svm.fit(X, y)
    ypred = svm.predict(Xte)
    return ypred

def run(args):
    loader = DataLoader()
    X0, Xte0, X1, Xte1, X2, Xte2 = get_kernels(args, loader)
    ytr0, ytr1, ytr2 = get_labels(loader)
    ypred0 = predict(X0, ytr0, 1.6e-3, Xte0)
    ypred1 = predict(X1, ytr1, 1.1e-3, Xte1)
    ypred2 = predict(X2, ytr2, .00081895, Xte2)

    list_preds = []
    for y_pred_test in [ypred0, ypred1, ypred2]:
        y_pred_test[y_pred_test == -1] = 0
        y_pred_test = y_pred_test.astype(int)
        list_preds += y_pred_test.tolist()

    with open("submission.csv", 'w') as f:
        f.write('Id,Bound\n')
        for i in range(len(list_preds)):
            f.write(str(i) + ',' + str(list_preds[i]) + '\n')

if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description='Run scripts for the MVA Kernel Methods Kaggle')
    argparser.add_argument('--precompute', action='store_true', help='enable early_stopping')
    run(argparser.parse_args())