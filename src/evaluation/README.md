# Evaluation utilities

## Metrics

Scoring metrics (see [here](https://en.wikipedia.org/wiki/Confusion_matrix) for detailed definitions):
* `accuracy_score`
* `precision_score`
* `recall_score`
* `specificity_score`
* `fpr_score`
* `auc_score`

Others:
* `roc_curve`
* `confusion_matrix`



## Selection

Provides cross validation utility which can be driven by any of the previous metrics.

Cross validation must be provided with :

- `clf`: a classifier instance
- `gram_matrix`: precomputed Gram matrix of the training set
- `ytrue`: groundtruth labels
- `cv`: number of cross-validation folds
- `scoring`: metric name to use
