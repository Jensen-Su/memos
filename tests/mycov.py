import numpy as np

def cov(X):
    shape = np.shape(X)
    n = shape[-1]
    e = np.ones(n)
    nex = np.dot(X,e)
    return (np.dot(X, X.T) - np.outer(nex, nex)/n)/(n-1)

