from costs import *
import numpy as np

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    N = x.shape[0]
    D = x.shape[1]
    tphi = np.ones((N, degree * D + 1))
    for j in range(degree):
        tphi[:,D*j+1:D*(j+1)+1] = x ** (j+1)
    return tphi


def ridge_regression(y, tx, lamb):
    """implement ridge regression."""
    m = tx.shape[1]
    im = np.zeros((m, m))
    np.fill_diagonal(im, 1)
    w_ridge = np.linalg.inv(tx.T.dot(tx) + lamb * 2 * len(y) * im).dot(tx.T).dot(y)
    mse = compute_loss(y, tx, w_ridge)
    return mse, w_ridge