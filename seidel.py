#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 10:59:49 2024

@author: nastyabekesheva
"""


import numpy as np

def simple_seidel(U, Ax, Ay, Cx, Cy, B, D, nx, ny, eps=1e-4, max_iter=10000):
    err = 1
    iter = 0
    while err > eps and iter < max_iter:
        iter += 1
        #if iter % 100 == 0:
            #print(f'iter = {iter:5d}; err = {err:4.3e}')
        err = 0
        U_old = U.copy()
        U[1:-1, 1:-1] = (Ax[1:-1, 1:-1] * U_old[:-2, 1:-1] + Cx[1:-1, 1:-1] * U_old[2:, 1:-1] +
                         Ay[1:-1, 1:-1] * U_old[1:-1, :-2] + Cy[1:-1, 1:-1] * U_old[1:-1, 2:] - D[1:-1, 1:-1]) / B[1:-1, 1:-1]
        err = np.max(np.abs(U - U_old))
    return U, iter

