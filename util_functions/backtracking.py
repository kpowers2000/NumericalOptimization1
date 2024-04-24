import numpy as np


def backtracking(f, d, x, rho, c):
    alphak = 1.0
    fk, gk = f(x)
    xx = x.copy()
    x = xx + alphak * d
    fk1 = f(x)[0]
    while fk1 > fk + c * alphak * np.dot(gk, d):
        alphak *= rho
        x = xx + alphak * d
        fk1 = f(x)[0]
    return alphak
