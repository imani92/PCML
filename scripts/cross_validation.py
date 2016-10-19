import numpy as np
from costs import *
from ridge_regression import *

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)



def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    y_te = y[k_indices[k]]
    y_tr = y[np.setxor1d(k_indices, k_indices[k])]
    x_te = x[k_indices[k]]
    x_tr = x[np.setxor1d(k_indices, k_indices[k])]
    
    phi_te = build_poly(x_te, degree)
    phi_tr = build_poly(x_tr, degree)
    
    # calculate weight and mse
    mse_tr, w_tr = ridge_regression(y_tr, phi_tr, lambda_)
    mse_te = compute_loss(y_te, phi_te, w_tr)
    
    # loss is rmse
    #loss_tr = (mse_tr * 2) ** 0.5
    #loss_te = (mse_te * 2) ** 0.5

    return mse_tr, mse_te, w_tr
