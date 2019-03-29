# Kernel based Classifiers

See [wiki on kernel classifiers](https://github.com/shahineb/kernel_challenge/wiki/Classifier-manipulation)

## Kernel Logistic Regression

Implementation of Kernel Logistic Regression based on IRLS updates.

## Kernel SVM

Implementation of SVM kernel solver based on dual formulation given by :

![caption](https://github.com/shahineb/kernel_challenge/blob/shahine/docs/svg/188bef82edb59df5b9234f60942a41c2.png)

Performed with `cvxopt` quadratic program solver. When training is done, non-supporting vectors are discarded for further computation.

## Kernel 2 SVM

Implementation of squared hinge loss classification problem based on dual formulation given by :

![caption](https://github.com/shahineb/kernel_challenge/blob/shahine/docs/svg/dual_square_hinge.png)

as well performed with `cvxopt` quadratic solver

## Multiple Kernel Learning

Implementation of [Multiple Kernel Learning with reduced gradient method](http://www.jmlr.org/papers/volume9/rakotomamonjy08a/rakotomamonjy08a.pdf)

Yields optimal convex combination of a list of kernels. Only works with SVM and Squared Hinge Loss classifiers.
