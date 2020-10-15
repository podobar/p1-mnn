import numpy as np
import math


class NeuralNetwork:

    Layers = list()
    Weights = list()
    Biases = list()
    Errors = list()

    def __init__(self, n, sizes):

        for i in range(len(sizes)-1):
            weights = np.random.rand(sizes[i+1], sizes[i])
            self.Weights.append(weights)

            bias = np.random.rand(sizes[i+1])
            self.Biases.append(bias)

    def forward_propagation(self, input, activate):
        self.Layers.append(input)

        for i in range(len(self.Weights)):
            new_input = self.Weights[i].dot(self.Layers[i])
            np.add(new_input, self.Biases[i])
            for j in range(len(new_input)):
                new_input[j] = activate(new_input[j])
            self.Layers.append(new_input)

        return self.Layers[-1]

    def back_propagation_error(self, expected, derivative, C):
        self.Errors.append(C(expected, self.Layers[-1]))

        for i in reversed(range(1, len(self.Layers)-1)):
            errors = list()
            for j in range(len(self.Layers[i])):
                error = 0
                for k in range(len(self.Layers[i+1])):
                    error += self.Weights[i][k][j] * self.Errors[-1][k]
                errors.append(error)
            for j in range(len(errors)):
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

    @staticmethod
    def sigmoid(x):
        return math.exp(x)/(math.exp(x) + 1)

    @staticmethod
    def sigmoid_derivative(y):
        return y(1-y)

    @staticmethod
    def subtract_cost_function(expected, result):
        return np.subtract(expected, result)


if __name__ == "__main__":

    network = NeuralNetwork(3, [2, 3, 1])
    result = network.forward_propagation([0, 1], NeuralNetwork.sigmoid)
    print(result)




