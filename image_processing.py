#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 11:10:13 2024

@author: nastyabekesheva
"""

import numpy as np 
from seidel import simple_seidel

def screened_poisson(Lambda, f):
    nx, ny = f.shape
    nx = nx - 2
    ny = ny - 2
    Ax = np.ones((nx+2, ny+2))
    Cx = np.ones((nx+2, ny+2))
    Ay = np.ones((nx+2, ny+2))
    Cy = np.ones((nx+2, ny+2))
    B = Ax + Cx + Ay + Cy + Lambda
    D = np.zeros((nx+2, ny+2))
    
    for i in range(1, nx+1):
        for j in range(1, ny+1):
            D[i, j] = (f[i-1, j] - 2*f[i, j] + f[i+1, j]) + (f[i, j-1] - 2*f[i, j] + f[i, j+1])
    
    U = np.zeros((nx+2, ny+2))
    j = np.arange(1, ny+1)
    B[1, j] = B[1, j] - Ax[1, j]
    Ax[1, j] = 0
    B[nx, j] = B[nx, j] - Cx[nx, j]
    Cx[nx, j] = 0
    
    i = np.arange(1, nx+1)
    B[i, 1] = B[i, 1] - Ay[i, 1]
    Ay[i, 1] = 0
    B[i, ny] = B[i, ny] - Cy[i, ny]
    Cy[i, ny] = 0
    
    U, _ = simple_seidel(U, Ax, Ay, Cx, Cy, B, D, nx, ny)
    
    U[0, j] = U[1, j]
    U[nx+1, j] = U[nx, j]
    U[i, 0] = U[i, 1]
    U[i, ny+1] = U[i, ny]
    
    U[0, 0] = (U[0, 1] + U[1, 0]) / 2
    U[nx+1, ny+1] = (U[nx, ny+1] + U[nx+1, ny]) / 2
    U[0, ny+1] = (U[0, ny] + U[1, ny+1]) / 2
    U[nx+1, 0] = (U[nx, 0] + U[nx+1, 1]) / 2
    
    maxU = np.max(U)
    minU = np.min(U)
    U = (U - minU) / (maxU - minU)
    
    return U