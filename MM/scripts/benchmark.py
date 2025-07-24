import numpy as np

EPS = 1e-6 # TODO: adjust eps 
# APE
def ape(y_hat, y): 
    if abs(y_hat - y) < EPS:
        return 0 
    return 2 * (y_hat - y) / (y_hat + y)

# GEH
def geh(c, m): 
    return np.sqrt(2*(m-c)*(m-c) / (m+c))

# MAE 
def mae(y_hat, y): 
    return abs(y_hat - y)

# RMSE
def rmse(y_hat, y): 
    return (y_hat - y)**2

