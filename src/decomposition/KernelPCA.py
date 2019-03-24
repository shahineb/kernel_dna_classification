import numpy as np



class KernelPCA():
    """
    Implementation of the kernel PCA

    Args:
        nb_components (int): number of components of the PCA
    """

    def __init__(self, nb_components=2):
        self.nb_components = nb_components
        
    def fit(self, gram_matrix):
        self.gram_matrix = gram_matrix
        # centering
        n = np.shape(gram_matrix)[0]
        I, U = np.eye(n), (1/n)*np.ones((n, n))
        centered_gram_matrix = (I - U).dot(gram_matrix).dot(I-U)
        # fitting
        self.eigval, self.eigvec = np.linalg.eig(centered_gram_matrix)
        ordering =  np.argsort(-self.eigval)
        self.eigval, self.eigvec = self.eigval[ordering], self.eigvec[:,ordering]
        self.components = self.eigvec[:,:self.nb_components] / np.sqrt(self.eigval[:self.nb_components])
        
    def transform(self, M):
        # Projects M of size (p, n) on the principal components of gram_matrix (n, n)
        p, n = M.shape
        I, U, V = np.eye(n), (1/n)*np.ones((p, n)), (1/n)*np.ones((n, n)), 
        centered_M = (M - U.dot(self.gram_matrix)).dot(I - V)
        return np.matmul(centered_M, self.components)