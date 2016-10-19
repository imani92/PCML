from costs import *
import numpy as np

def sigma(x):
    sigma = float(1) / (1 + np.exp(-x))
    return sigma

def compute_logistic_loss(y, tx, w):
    total = np.sum(np.log(1 + np.exp(tx.dot(w)))) - y.T.dot(tx.dot(w))
    average = total / len(y)
    return average

def compute_logistic_gradient(y, tx, w):
    grad = tx.T.dot(sigma(tx.dot(w)) - y)
    return grad
    
def logistic_regression(y, tx, max_iters, gamma):
    initial_w = np.array([1.0] * tx.shape[1])
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_logistic_gradient(y, tx, w)
        loss = compute_logistic_loss(y, tx, w)
        w = w - gamma * grad
        if (np.remainder(n_iter, 10) == 0):
            print("logistic regression with GD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                    bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return loss, w

def predict_logistic_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0.5)] = -1
    y_pred[np.where(y_pred > 0.5)] = 1
    
    return y_pred