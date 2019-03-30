import numpy as np
import scipy.stats
import scipy as sp
import random


class EM_HMM():

    def __init__(self, k=4, tol=1e-2):
        '''
        Attributes:
        self.tol: tolerance criterion for convergence
        self.pi_0: parameter vector for the multinomial latent variable q_0
        self.mu: (k, p) array of the k centroids
        self.sigmas = (k, p ,p) array of the k covariance matrices
        self.a = (k, k) array corresponding to the transition matrix between q_t and q_t+1
        '''
        self.states = k
        self.tol = tol
        self.pi_0 = np.ones(self.states) / self.states
        self.mu = None
        self.sigma = None
        self.a = (1/self.states)*np.ones((self.states, self.states))

    @staticmethod
    def stable_log_sum(log_vector):
        log_max = np.max(log_vector)
        idx_max = np.argmax(log_vector)
        log_vector = log_vector - log_max
        log_vector = np.hstack((log_vector[:idx_max], log_vector[idx_max + 1:]))
        return log_max + np.log(1 + np.sum(np.exp(log_vector), dtype=np.longdouble), dtype=np.longdouble)

    @staticmethod
    def stable_log_message_passing_alpha(a_q_t, log_alpha_tm1):
        return EM_HMM.stable_log_sum(np.log(a_q_t) + log_alpha_tm1)

    def build_log_alpha(self, X):

        # Define arrays for log values of alpha
        log_alpha, log_alpha_t = np.zeros(self.states), np.zeros(self.states)

        # Initialization
        for q in range(self.states):
            log_alpha[q] = np.log(self.pi_0[q]) + sp.stats.multivariate_normal.logpdf(X[0], self.mu[q], self.sigma[q])
        log_alpha = np.expand_dims(log_alpha, 0)

        # Recursion for t=1,...,T
        for t in range(1, X.shape[0]):
            for q in range(self.states):
                log_alpha_t[q] = sp.stats.multivariate_normal.logpdf(X[t], self.mu[q], self.sigma[q]) + \
                                 EM_HMM.stable_log_message_passing_alpha(self.a[:, q], log_alpha[-1])
            log_alpha = np.vstack((log_alpha, log_alpha_t))

        return log_alpha

    def stable_log_message_passing_beta(self, a_q_t, log_beta_tp1, X_tp1):
        log_pdf = np.zeros(self.states)
        for q_tp1 in range(self.states):
            log_pdf[q_tp1] = sp.stats.multivariate_normal.logpdf(X_tp1, self.mu[q_tp1], self.sigma[q_tp1])

        return EM_HMM.stable_log_sum(np.log(a_q_t) + log_pdf + log_beta_tp1)

    def build_log_beta(self, X):
        # Define arrays for log values of beta
        log_beta, log_beta_t = np.zeros((1, self.states)), np.zeros(self.states)

        # Recursion for t=T-2,...,0
        for t in range(X.shape[0] - 2, -1, -1):
            for q in range(self.states):
                log_beta_t[q] = self.stable_log_message_passing_beta(self.a[q], log_beta[-1], X[t + 1])
            log_beta = np.vstack((log_beta, log_beta_t))

        return log_beta[::-1]  # we reverse the order to get values from t=0 to t=T

    @staticmethod
    def unary_prob(log_alpha, log_beta):
        # returns a T x K array s.t. array[t,k] = p(q_t=k|uo,...,uT)
        log_prob = np.zeros(log_alpha.shape)
        for t in range(log_alpha.shape[0]):
            log_Z = EM_HMM.stable_log_sum(log_alpha[t] + log_beta[t])
            log_prob[t] = log_alpha[t] + log_beta[t] - log_Z
        return np.exp(log_prob)

    def pairwise_prob(self, log_alpha, log_beta, X):
        # returns a T x K x K array s.t. array[t,k,l] = p(q_t=k, q_t+1=l|uo,...,uT)
        log_prob_pairwise = np.zeros((X.shape[0] - 1, self.states, self.states))
        for t in range(X.shape[0] - 1):
            for q_t in range(4):
                for q_tp1 in range(4):
                    log_prob_pairwise[t, q_t, q_tp1] = log_alpha[t, q_t] + log_beta[t + 1, q_tp1] + np.log(
                        self.a[q_t, q_tp1]) + sp.stats.multivariate_normal.logpdf(X[t + 1], self.mu[q_tp1], self.sigma[q_tp1])
            log_prob_pairwise[t, :, :] -= EM_HMM.stable_log_sum(np.ndarray.flatten(log_prob_pairwise[t, :, :]))
        return np.exp(log_prob_pairwise)

    def fit(self, X, max_iter=100):
        '''
        X: (n, p) np.array data matrix

        Returns:
        self: fitted parameters
        '''

        # initialization of mu and sigmas with GMM results
        self.mu = np.zeros(self.states)
        self.sigma = np.eye(self.states)

        # convergence criteria
        it = 0
        conv = False
        log_likelihood_1 = self.log_likelihood_hmm(X)
        log_likelihood_2 = log_likelihood_1.copy()

        while not conv:

            # compute forward and backward messages
            log_alpha = self.build_log_alpha(X)
            log_beta = self.build_log_beta(X)

            # compute probabilities for latent variables (E-step)
            prob_unary = EM_HMM.unary_prob(log_alpha, log_beta)
            prob_pairwise = self.pairwise_prob(log_alpha, log_beta, X)

            # update pi_0, a, mus and sigmas to maximize likelihood (M-step)
            self.pi_0 = prob_unary[0] / np.sum(prob_unary[0])

            self.a = np.sum(prob_pairwise, axis=0) / np.sum(prob_unary[:-1], axis=0)[:, None]

            for c in range(self.states):
                self.mu[c] = np.matmul(prob_unary[:, c], X) / np.sum(prob_unary[:, c])
                self.sigma[c] = np.matmul((X - self.mu[c]).T,
                                           np.expand_dims(prob_unary[:, c], axis=1) * (X - self.mu[c])) / np.sum(
                    prob_unary[:, c])

            it += 1
            log_likelihood_1 = log_likelihood_2
            log_likelihood_2 = self.log_likelihood_hmm(X)
            criterion = abs((log_likelihood_2 - log_likelihood_1) / log_likelihood_1)

            if it > max_iter or criterion < self.tol:
                conv = True
                print("Iteration {} - |(LL1-LL2)/LL1|={}".format(it, criterion))

    def predict(self, X):
        '''
        X: (n, p) np.array data matrix

        Returns:
        q: most probable latent variables
        '''

        # compute forward and backward messages
        log_alpha = self.build_log_alpha(X)
        log_beta = self.build_log_beta(X)

        # compute probabilities for latent variables (E-step)
        prob_unary = EM_HMM.unary_prob(log_alpha, log_beta)

        return np.argmax(prob_unary, axis=1)

    def log_likelihood_hmm(self, X):
        q = self.predict(X)
        likelihood = np.log(self.pi_0[q[0]])
        n = X.shape[0]
        for i in range(n - 1):
            likelihood += np.log(self.a[q[i], q[i + 1]]) + sp.stats.multivariate_normal.logpdf(X[i], self.mu[q[i]], self.sigma[q[i]])

        likelihood += sp.stats.multivariate_normal.logpdf(X[n - 1], self.mu[q[n - 1]], self.sigma[q[n - 1]])

        return likelihood

