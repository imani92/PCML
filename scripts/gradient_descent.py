# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Gradient Descent
"""
import numpy as np
from costs import *


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y - tx.dot(w)
    grad = - tx.T.dot(e) / y.shape[0]
    return grad


def gradient_descent(y, tx, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    initial_w = np.array([0.0] * tx.shape[1])
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient and loss
        grad = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        # update w by gradient
        w = w - gamma * grad
        if (np.remainder(n_iter, 10) == 0):
            print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                    bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return loss, w
