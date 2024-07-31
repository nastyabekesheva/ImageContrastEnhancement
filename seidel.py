#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 10:59:49 2024

@author: nastyabekesheva
"""


import numpy as np

def simple_seidel(U, Ax, Ay, Cx, Cy, B, D, nx, ny, eps=1e-4):
    err = 1
    iter = 0
    while err > eps:
        iter += 1
        if iter % 100 == 0:
            print(f'iter = {iter:5d}; err = {err:4.3e}')
        err = 0
        for i in range(1, nx+1):
            for j in range(1, ny+1):
                t = (Ax[i,j]*U[i-1,j] + Cx[i,j]*U[i+1,j] +
                     Ay[i,j]*U[i,j-1] + Cy[i,j]*U[i,j+1] - D[i,j]) / B[i,j]
                div = abs(t - U[i,j])
                if err < div:
                    err = div
                U[i,j] = t
    return U, iter


