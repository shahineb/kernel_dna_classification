from sklearn import cross_validation, metrics
import numpy as np
import matplotlib.pyplot as plt
import logging
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

NFOLDS = 10


class Sklearner:

    """
    A class to help us through the training process

    Attributes :
        clf : classifier
        param_grid : in case we want to perform a grid search
    """

    def __init__(self, clf, Xtr, ytr, Xval, yval, param_grid=None):
        self.clf = clf
        self.param_grid = param_grid
        self.Xtr = Xtr
        self.ytr = ytr
        self.Xval = Xval
        self.yval = yval
        self.pred = np.zeros(len(yval))
        self.pred_proba = np.zeros(len(yval))
        self.crossValidated = False
        self.auc_score = 0

    def train(self):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        self.clf.fit(self.Xtr, self.ytr)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def feature_importances(self):
        try:
            return self.clf.fit(self.Xtr, self.ytr).feature_importances_
        except AttributeError as e:
            print(e)

    def cross_validate(self):
        # generate the cross_validation folds
        cv_folds = cross_validation.StratifiedKFold(self.yval, NFOLDS, shuffle=True)

        for i, (tr, te) in enumerate(cv_folds):
            # restrict data to the local training/testing set
            Xtr_ = self.Xval[tr, :]
            ytr_ = self.yval[tr]
            Xte_ = self.Xval[te, :]

            # fit classifier
            self.clf.fit(Xtr_, ytr_)

            # predict
            yte_pred = self.predict(Xte_)
            yte_pred_proba = self.predict_proba(Xte_)
            self.pred[te] = yte_pred
            self.pred_proba[te] = yte_pred_proba
        self.crossValidated = True
        return None

    def roc_score(self):
        """
        Computes the area under roc curves and plots roc curve
        :return:
        """
        if self.crossValidated:
            fpr, tpr, thresholds = metrics.roc_curve(y_true=self.yval.values, y_score=self.pred)
            self.auc_score = metrics.auc(fpr, tpr)
            plt.plot(fpr, tpr, '-', color='tomato', label='AUC = %0.3f' % self.auc_score)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC curve')
            plt.grid(alpha=0.5)
            plt.legend(loc="lower right")
        else:
            print("Must cross_validate first")
        return None

    def predictionsAccuracy(self):
        """
        plots distribution of predictions' probabilities
        :return:
        """
        if self.crossValidated:
            sns.plt.figure(figsize=(10, 6))
            sns.plt.title("Distribution of predictions' probability")
            sns.distplot(self.pred_proba, kde=True, bins=30, color='chocolate')
        else:
            print("Must cross_validate first")
