import os
import sys
import numpy as np

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from src.evaluation.metrics import *


disct_scoring_metrics = {"accuracy_score": accuracy_score,
                         "precision_score": precision_score,
                         "recall_score": recall_score,
                         "specificity_score": specificity_score,
                         "fpr_score": fpr_score,
                         "auc_score": auc_score}


def cross_validate(classifier, gram_matrix, ytrue, cv=5, scoring="accuracy_score"):
    """
    Args:
        clf (Classifier): classifier object
        gram_matrix (np.ndarray): precomputed Gram matrix of the training set
        ytrue (np.ndarray): ground-truth labeling of training set
        cv (int): number of folders for cross validation
        scoring (str): Evaluation metric (one of "accuracy_score", "precision_score",
                        "recall_score", "specificity_score", "fpr_score", "auc_score")
    Returns:
        scores (list): evaluation score on each folder

    """

    n = np.shape(gram_matrix)[0]
    idxes = np.arange(n)
    np.random.shuffle(idxes)
    scores = []
    score_function = disct_scoring_metrics[scoring]

    for f in range(cv):
        ids = np.arange(f*n//cv, (f+1)*n//cv)
        idxes_te = idxes[ids]
        idxes_tr = np.array( list( set(idxes).difference(set(idxes_te)) ) )

        gram_matrix_tr = gram_matrix[idxes_tr[:, None], idxes_tr]
        gram_matrix_te = gram_matrix[idxes_tr[:, None], idxes_te]
        ytr, yte = ytrue[idxes_tr] , ytrue[idxes_te]

        classifier.fit(gram_matrix_tr, ytr, precomputed=True)
        ypred = classifier.predict(gram_matrix_te, precomputed=True)
        scores.append( score_function(yte, ypred) )

    print("Mean accuracy = {} - Std = {}".format(np.mean(scores), np.std(scores)))

    return scores
