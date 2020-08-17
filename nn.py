#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
from planar_utils import load_planar_dataset
import h5py
from dnn_app_utils_v3 import *


class NN:

    def __init__(self, samples=10, pixels=100, layers=2,
                 w=0, b=0, z=0, a=0, activations=0, dw=0,
                 db=0, dz=0, da=0):

        self.layers = layers
        self.samples = samples
        self.pixels = pixels
        self.w = [w]
        self.b = [b]
        self.z = [z]
        self.a = [a]
        self.activation = [activations]
        self.dw = [dw]
        self.db = [db]
        self.dz = [dz]
        self.da = [da]
        __all__ = [np.dot, np.zeros, np.tanh, np.exp]

    def create_network(self, x, layers=2, layer_details=[('sigmoid', 4), ('sigmoid', 1)]):

        samples = x.shape[1]  # Number of training examples
        pixels = x.shape[0]  # Number of pixels in one example
        w = [0]*(layers+1)  # Weights Matrix
        dw = [0]*(layers+1)  # Weights Grad Matrix
        b = [0]*(layers+1)  # Bias Matrix
        db = [0]*(layers+1)  # Bias Grad Matrix
        z = [0]*(layers+1)  #
        dz = [0]*(layers+1)  #
        a = [0]*(layers+1)  #
        da = [0]*(layers+1)  #
        activations = ['0']*(layers+1)  # Activation functions of layers
        a[0] = x  # Layer 0 stores the input data

        np.random.seed(1)
        for i, (n, v) in enumerate(layer_details):
            w[i+1] = (np.random.randn(v, pixels)) * 0.01
            b[i+1] = (np.zeros((v, 1)))
            z[i+1] = (np.zeros((v, 1)))
            a[i+1] = (np.zeros((v, 1)))
            activations[i+1] = (str(n))
            pixels = v

        self.layers = layers
        self.samples = samples
        self.pixels = pixels
        self.w = w
        self.b = b
        self.z = z
        self.a = a
        self.activation = activations
        self.dw = dw
        self.db = db
        self.dz = dz
        self.da = da

    # Function that performs the forward propogation in our neural network
    def forward_pass(self):

        for l in range(self.layers):
            self.z[l+1] = np.dot(self.w[l+1], self.a[l]) + self.b[l+1]

            if self.activation[l+1] == 'sigmoid':
                self.a[l+1] = 1 / (1 + np.exp(-self.z[l+1]))
            if self.activation[l+1] == 'tanh':
                self.a[l+1] = np.tanh(self.z[l+1])
            if self.activation[l + 1] == 'relu':
                self.a[l+1] = np.maximum(0, self.z[l+1])

    def backward_pass(self, Y):

        self.da[self.layers] = - (np.divide(Y, self.a[self.layers])
                                  - np.divide(1 - Y, 1 - self.a[self.layers]))

        for l in range(self.layers, 0, -1):
            if self.activation[l] == "sigmoid":
                self.dz[l] = (sigmoid_backward(self.da[l], self.z[l]))
            if self.activation[l] == "relu":
                self.dz[l] = (relu_backward(self.da[l], self.z[l]))

            m = self.a[l-1].shape[1]
            self.dw[l] = np.dot(self.dz[l], self.a[l-1].T) * 1./m
            self.db[l] = np.sum(self.dz[l], axis=1, keepdims=True) * 1./m
            self.da[l-1] = np.dot(self.w[l].T, self.dz[l])

    def update_weights(self, learning_rate=0.05):

        for l in range(self.layers):
            self.w[l+1] = self.w[l+1] - (learning_rate * self.dw[l+1])
            self.b[l+1] = self.b[l+1] - (learning_rate * self.db[l+1])

    def predict(self):

        prediction = (self.a > 0.5)
        return prediction

    def model(self, X, Y, classes=2, iterations=1000, learning_rate=0.05):

        # for i in range(classes):
        #     Y = Y[Y == i]
        #     print(Y.shape)
        for _ in range(iterations):
            self.forward_pass()
            # if _ == 0:
            #     print("a1:" + str(self.a[2]))
            #     # print("a2:" + str(self.a[1]))
            #     print("z1:" + str(self.z[2]))
            #     print("w1:" + str(self.w[2]))
            cost = compute_cost(self.a[self.layers], Y)
            if _ % 100 == 0:
                print("Cost after iteration %i: %f" % (_, cost))
            self.backward_pass(Y)
            self.update_weights(learning_rate)

        predicted = self.predict()

    def check_network(self):

        for i in range(self.layers):
            print("\n")
            print("For layer number "+str(i+1)+":")
            print("w:"+str(self.w[i+1].shape))
            print("b:"+str(self.b[i+1].shape))
            print("z:"+str(self.z[i+1].shape))
            print("a:"+str(self.a[i+1].shape))
            print("Activation Function:"+str(self.activation[i+1]))


def sigmoid_backward(act, zee):

    s = 1 / (1 + np.exp(-zee))
    zee = act * s * (1 - s)

    return zee


def relu_backward(dA, cache):

    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ


def compute_cost(al, y):

    m = y.shape[1]
    cost = (1. / m) * (-np.dot(y, np.log(al).T) - np.dot(1 - y, np.log(1 - al).T))

    cost = np.squeeze(cost)

    return cost


if __name__ == "__main__":

    np.random.seed(1)
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

    # Reshape the training and test examples
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0],
                                           -1).T  # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten / 255.
    test_x = test_x_flatten / 255.

    details = [('relu', 7), ('sigmoid', 1)]

    nn_model = NN()
    nn_model.create_network(train_x, 2, details)
    nn_model.check_network()
    nn_model.model(train_x, train_y, 2, 2500, 0.0075)
