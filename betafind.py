# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 17:55:09 2019

@author: Will Clare
"""

import numpy as np

def beta(r,r1,eps):
    b = (r**2+r1**2)/(r*r1) + eps**2
    B = 1/2*(b - (b**2 - 4)**(1/2))
    return B

def beta1(r,eps):
    n = len(r)
    B = np.zeros(shape=(n,n))
    for i in range(n):
        for j in range(n):
            B[i,j] = beta(r[i],r[j],eps)
    return B