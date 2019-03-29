import numpy as np
import matplotlib.pyplot as plt

disct_scoring_metrics = dict()


def catalog(metric_name):
    """Workaround wrapper to map each metric to a string
    Args:
        name (str)
    """
    global disct_scoring_metrics

    def wrapper(func):
        func._marker = metric_name
        disct_scoring_metrics[metric_name] = func
        return func
    return wrapper


###############################################################################


@catalog("accuracy_score")
def accuracy_score(ytrue, ypred):
    """Accuracy metric.
    Args:
        ytrue (np.array): true labeling
        ypred (np.array): predicted labeling
    """
    return np.sum(ytrue == ypred) / len(ytrue)


@catalog("precision_score")
def precision_score(ytrue, ypred):
    """Precision metric.
    Args:
        ytrue (np.array): true labeling
        ypred (np.array): predicted labeling
    """
    fp = np.sum((ytrue != ypred) & (ytrue == -1))
    tp = np.sum((ytrue == ypred) & (ytrue == 1))
    return tp / (tp + fp)


@catalog("recall_score")
def recall_score(ytrue, ypred):
    """Recall metric.
    Args:
        ytrue (np.array): true labeling
        ypred (np.array): predicted labeling
    """
    fn = np.sum((ytrue != ypred) & (ytrue == 1))
    tp = np.sum((ytrue == ypred) & (ytrue == 1))
    return tp / (tp + fn)


@catalog("specificity_score")
def specificity_score(ytrue, ypred):
    """Specificity metric.
    Args:
        ytrue (np.array): true labeling
        ypred (np.array): predicted labeling
    """
    tn = np.sum((ytrue == ypred) & (ytrue == -1))
    fp = np.sum((ytrue != ypred) & (ytrue == -1))
    return tn / (tn + fp)


@catalog("fpr_score")
def fpr_score(ytrue, ypred):
    """FPRy metric.
    Args:
        ytrue (np.array): true labeling
        ypred (np.array): predicted labeling
    """
    tn = np.sum((ytrue == ypred) & (ytrue == -1))
    fp = np.sum((ytrue != ypred) & (ytrue == -1))
    return fp / (tn + fp)


@catalog("confusion_matrix")
def confusion_matrix(ytrue, ypred, plot=True, title='Confusion matrix'):
    """Accuracy metric.
    Args:
        ytrue (np.array): true labeling
        ypred (np.array): predicted labeling
    """
    tn = np.sum((ytrue == ypred) & (ytrue == -1))
    fn = np.sum((ytrue != ypred) & (ytrue == 1))
    fp = np.sum((ytrue != ypred) & (ytrue == -1))
    tp = np.sum((ytrue == ypred) & (ytrue == 1))
    cm = np.array([[tn, fp], [fn, tp]])

    def plot_confusion_matrix(cm):
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        plt.xticks(np.array([0, 1]), ['0', '1'])
        plt.yticks(np.array([0, 1]), ['0', '1'])
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    if plot:
        plot_confusion_matrix(cm)
    else:
        return cm


@catalog("roc_curve")
def roc_curve(ytrue, ypred):
    """ROC curve.
    Args:
        ytrue (np.array): true labeling
        ypred (np.array): predicted scores
    Returns:
        fpr (list): list of False Positive Rates
        tpr (list): list of True Positive Rates
    """
    fpr, tpr = [], []
    idx_sort = np.argsort(ypred)
    ytrue, ypred = ytrue[idx_sort], ypred[idx_sort]
    for t in ypred:
        ypred_t = np.sign(ypred - t)
        fpr.append(fpr_score(ytrue, ypred_t))
        tpr.append(recall_score(ytrue, ypred_t))
    return fpr, tpr


@catalog("auc_score")
def auc_score(fpr, tpr):
    """AUC score.
    Args:
        fpr (list): list of False Positive Rates
        tpr (list): list of True Positive Rates
    Returns:
        auc (float): area under ROC curve
    """
    return -np.trapz(tpr, fpr)
