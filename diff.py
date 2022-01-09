# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 12:23:19 2019

@author: Will Clare
"""

import numpy as np

def diff(f,i,k,X):
    if len(np.shape(f)) == 2:
        d = (f[i+1,k] - f[i,k])/(X[i+1]-X[i])
    else:
        d = (f[i+1] - f[i])/(X[i+1]-X[i])
    return d

def diff2(f,i,k,X):
    dX = X[2] - X[1]
    d = (f[i+1,k] - 2*f[i,k] + f[i-1,k])/dX**2
    return d

def diffm(f,k,X):
    d = np.zeros(len(X))
    for i in range(len(X)-1):
        d[i] = diff(f,i,k,X)
    d[len(X)-1] = d[len(X)-2]
    return d

def difm(f,X):
    d = np.zeros(shape=(np.shape(f)))
    for k in range(len(f[0,:])):
        for i in range(len(X)-1):
            d[i,k] = diff(f,i,k,X)
        d[len(X)-1,k] = d[len(X)-2,k]
    return d