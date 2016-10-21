from costs import *
from logistic_regression import *
from stochastic_gradient_descent import *
import numpy as np

def compute_hessian(y, tx, w):
    """return the hessian of the loss function."""
    diag = sigmoid(tx.dot(w)) * (1-sigmoid(tx.dot(w)))
    S = np.diag(diag)
    H = (tx.T.dot(S)).dot(tx)
    return H

def logistic_regression(y, tx, w):
    """return the loss, gradient, and hessian."""
    loss = compute_logistic_loss(y, tx, w)
    grad = compute_logistic_gradient(y, tx, w)
    H = compute_hessian(y, tx, w)
    return loss, grad, H

def learning_by_newton_method(y, tx, w, gamma, regularized=False, lambda_=0.0):
    """
    Do one step on Newton's method.
    return the loss and updated w.
    """
    loss, grad, H = logistic_regression(y, tx, w)
    if (regularized == True):
        reg_grad = 2 * lambda_ * w
        reg_H = 2 * np.identity(len(w))
        H += reg_H
        grad += reg_grad
        
    w = w - np.linalg.inv(H).dot(grad) * gamma
    return loss, w

def newton_method(y, tx, batch_size, max_iters, gamma, regularized=False, lambda_=0.0):
    initial_w = np.array([0.5] * tx.shape[1])
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            loss, w = learning_by_newton_method(minibatch_y, minibatch_tx, w, gamma, regularized, lambda_)
        if (np.remainder(n_iter, 50) == 0):
            print("logistic with Newton({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                    bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return loss, w