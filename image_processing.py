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
    nx -= 2
    ny -= 2
    Ax = np.ones((nx + 2, ny + 2))
    Cx = np.ones((nx + 2, ny + 2))
    Ay = np.ones((nx + 2, ny + 2))
    Cy = np.ones((nx + 2, ny + 2))
    B = Ax + Cx + Ay + Cy + Lambda
    D = np.zeros((nx + 2, ny + 2))
    
    D[1:-1, 1:-1] = (f[:-2, 1:-1] - 2*f[1:-1, 1:-1] + f[2:, 1:-1]) + (f[1:-1, :-2] - 2*f[1:-1, 1:-1] + f[1:-1, 2:])
    
    U = np.zeros((nx + 2, ny + 2))
    B[1, 1:-1] -= Ax[1, 1:-1]
    Ax[1, 1:-1] = 0
    B[-2, 1:-1] -= Cx[-2, 1:-1]
    Cx[-2, 1:-1] = 0
    B[1:-1, 1] -= Ay[1:-1, 1]
    Ay[1:-1, 1] = 0
    B[1:-1, -2] -= Cy[1:-1, -2]
    Cy[1:-1, -2] = 0
    
    U, _ = simple_seidel(U, Ax, Ay, Cx, Cy, B, D, nx, ny)
    
    U[0, 1:-1] = U[1, 1:-1]
    U[-1, 1:-1] = U[-2, 1:-1]
    U[1:-1, 0] = U[1:-1, 1]
    U[1:-1, -1] = U[1:-1, -2]
    
    U[0, 0] = (U[0, 1] + U[1, 0]) / 2
    U[-1, -1] = (U[-2, -1] + U[-1, -2]) / 2
    U[0, -1] = (U[0, -2] + U[1, -1]) / 2
    U[-1, 0] = (U[-2, 0] + U[-1, 1]) / 2
    
    maxU = np.max(U)
    minU = np.min(U)
    U = (U - minU) / (maxU - minU)
    
    return U