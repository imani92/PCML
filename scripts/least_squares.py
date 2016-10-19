from costs import *

def least_squares(y, tx):
    """calculate the least squares solution."""
    w = np.linalg.inv(tx.T.dot(tx)).dot(tx.T).dot(y)
    mse = compute_loss(y, tx, w)
    return mse, w