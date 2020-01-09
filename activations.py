#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 11:25:08 2020

@author: milan
"""

import numpy as np

#activation function and derivative

def tanh(x):
    return(np.tanh(x));

def tanh_prime(x):
    return(1-np.tanh(x)**2);