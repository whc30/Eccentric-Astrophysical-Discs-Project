# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 12:55:02 2019

@author: Will Clare
"""

import numpy as np
import scipy.integrate as integrate
cos = np.cos
pi = np.pi


def LC(s,j,a):
    b = 1/pi*integrate.quad(lambda t: cos(j*t)/(1 - 2*a*cos(t) + a**2)**s,0,2*pi)[0]
    return b