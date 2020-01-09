#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 15:14:30 2020

@author: milan
"""

import numpy as np
from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activations import tanh, tanh_prime
from losses import mse, mse_prime

from keras.datasets import mnist
from keras.utils import np_utils

#load MNIST from server
(x_train, y_train),(x_test, y_test) = mnist.load_data()

#training data
#reshape and normalize input data
x_train = x_train.reshape(x_train.shape[0],1,28*28)
x_train = x_train.astype('float32')
x_train /= 255

#ecnoding output
y_train = np_utils.to_categorical(y_train)

#same for test data
x_test = x_test.reshape(x_test.shape[0],1,28*28)
x_test = x_test.astype('float32')
x_test /= 255

#ecnoding output
y_test = np_utils.to_categorical(y_test)

#Network
net = Network()
net.add(FCLayer(28*28, 50))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(50, 10))
net.add(ActivationLayer(tanh, tanh_prime))

net.use(mse, mse_prime)
net.fit(x_train[:1000], y_train[:1000],epochs=50,learning_rate=0.1)


#test on 3 samples
out = net.predict(x_test[:1])
print('\n')
print('true values: ')
print(y_test[0:1])
print('\n')
print('predicted values: ')

out_int = [np.round(x) for x in out]
print(out_int)