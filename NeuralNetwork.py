import numpy as np
import math
import csv
from Activation import Activation


class NeuralNetwork:

    Layers = list()
    Weights = list()
    Biases = list()
    Errors = list()

    def __init__(self, n, sizes):

        for i in range(len(sizes)-1):
            weights = np.random.rand(sizes[i+1], sizes[i])*np.sqrt(2/sizes[i])
            self.Weights.append(weights)

            bias = np.random.rand(sizes[i+1])
            self.Biases.append(bias)

    def forward_propagation(self, input, inner_activate, out_activate):
        self.Layers.clear()
        self.Layers.append(input)

        for i in range(len(self.Weights)):
            new_input = self.Weights[i].dot(self.Layers[i])
            np.add(new_input, self.Biases[i])
            for j in range(len(new_input)):
                f = out_activate if i == len(self.Weights)-1 else inner_activate
                new_input[j] = f(new_input[j])
            self.Layers.append(new_input)

        return self.Layers[-1]

    def back_propagation_error(self, expected, inner_derivative, out_derivative, cost_gradient):
        self.Errors.append(cost_gradient(expected, self.Layers[-1]))

        for i in range(len(self.Layers)-2, 0, -1):
            errors = list()
            for j in range(len(self.Layers[i])):
                error = 0
                for k in range(len(self.Layers[i+1])):
                    error += self.Weights[i][k][j] * self.Errors[-1][k]
                errors.append(error)
            for j in range(len(errors)):
                derivative = out_derivative if i == len(self.Layers)-2 else inner_derivative
                errors[j] *= derivative(self.Layers[i][j])
            self.Errors.append(errors)

        self.Errors.reverse()

    def update_weights(self, factor):
        for i in range(len(self.Weights)):
            for j in range(len(self.Layers[i+1])):
                for k in range(len(self.Layers[i])):
                    self.Weights[i][j][k] += factor * self.Errors[i][j] * self.Layers[i][k]
                self.Biases[i][j] += factor * self.Errors[i][j]
        self.Errors.clear()





