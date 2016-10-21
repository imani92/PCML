from costs import *
import numpy as np

def sigmoid(x):
    sigma = float(1) / (1 + np.exp(-x))
    return sigma

def compute_logistic_loss(y, tx, w):
    total = np.sum(np.log(1 + np.exp(tx.dot(w)))) - y.T.dot(tx.dot(w))
    average = total / len(y)
    return average

def compute_logistic_gradient(y, tx, w):
    grad = tx.T.dot(sigmoid(tx.dot(w)) - y)
    return grad

def learning_by_gradient_descent(y, tx, w, gamma, regularized=False, lambda_=0.0):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    loss = compute_logistic_loss(y, tx, w)
    grad = compute_logistic_gradient(y, tx, w)
    if (regularized == False):
        w = w - gamma * grad
    else:
        w = w - gamma * (grad + 2 * lambda_ * w)
        
    return loss, w
    
def logistic_regression(y, tx, max_iters, gamma, regularized=False, lambda_=0.0):
    initial_w = np.array([0.5] * tx.shape[1])
    w = initial_w
    for n_iter in range(max_iters):
        loss, w = learning_by_gradient_descent(y, tx, w, gamma, regularized, lambda_)
        if (np.remainder(n_iter, 20) == 0):
            print("logistic regression with GD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                    bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return loss, w


def predict_logistic_labels(weights, data):
    """Convert logistic predictions to standard Higgs boson predictions"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0.5)] = -1
    y_pred[np.where(y_pred > 0.5)] = 1
    
    return y_pred