#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 12:19:10 2020

@author: milan
"""

import numpy as np
from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activations import tanh, tanh_prime
from losses import mse, mse_prime

#training data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

#network
net = Network()
net.add(FCLayer(2,3))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(3, 1))
net.add(ActivationLayer(tanh, tanh_prime))

#train
net.use(mse, mse_prime)

net.fit(x_train, y_train, epochs=10000, learning_rate=0.1)

#test
out = net.predict(x_train)
print(out)   