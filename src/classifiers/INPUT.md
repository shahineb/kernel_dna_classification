# Kernel based Classifiers

## Kernel Logistic Regression

_TO COMPLETE_


## Kernel SVM

Implementation of SVM kernel solver based on dual formulation given by :

$$
\begin{aligned}
  & \underset{\alpha\in\mathbb{R}^n}{\min}
  & & \frac{1}{2}\alpha^\top\mathbf{K}\alpha - y^\top\alpha\\
  & \text{s.t.}
  & & 0 \leq y_i\alpha_i\leq \frac{1}{2\lambda n} \enspace \forall i\in[\![1,n]\!]
\end{aligned}
$$

Performed with `cvxopt` quadratic program solver. When training is done, non-supporting vectors are discarded for further computation.
