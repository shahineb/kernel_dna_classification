# Kernel based Classifiers

## Kernel Logistic Regression

_TO COMPLETE_


## Kernel SVM

Implementation of SVM kernel solver based on dual formulation given by :

<p align="center"><img alt="$$&#10;\begin{aligned}&#10;  &amp; \underset{\alpha\in\mathbb{R}^n}{\min}&#10;  &amp; &amp; \frac{1}{2}\alpha^\top\mathbf{K}\alpha - y^\top\alpha\\&#10;  &amp; \text{s.t.}&#10;  &amp; &amp; 0 \leq y_i\alpha_i\leq \frac{1}{2\lambda n} \enspace \forall i\in[\![1,n]\!]&#10;\end{aligned}&#10;$$" src="https://rawgit.com/shahineb/kernel_challenge/shahine/docs/svg//188bef82edb59df5b9234f60942a41c2.png?invert_in_darkmode" align="middle" width="244.69352984999998pt" height="74.46104654999999pt"/></p>

Performed with `cvxopt` quadratic program solver. When training is done, non-supporting vectors are discarded for further computation.
