from costs import *
from logistic_regression import *
import numpy as np

def s(tx, w):
    diag = sigma(tx.dot(w)) * (1-sigma(tx.dot(w)))
    return np.diag(diag)

def hessian(tx, s):
    H = tx.T.dot(s).dot(tx)
    return H

def newton_method(y, tx, max_iters, gamma):
    initial_w = np.array([1.0] * tx.shape[1])
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_logistic_gradient(y, tx, w)
        loss = compute_logistic_loss(y, tx, w)
        H = hessian(tx, s(tx, w))
        w = w - np.linalg.inv(H).dot(grad) * gamma
        print("logistic regression with Newton({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return loss, w