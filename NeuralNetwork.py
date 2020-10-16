import numpy as np
import math
import csv


def load_csv(filename):
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        _data = list(reader)

    for index in range(1, len(_data)):
        _data[index] = [float(item) for item in _data[index]]

    return _data


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
        self.Layers.clear()
        self.Layers.append(input)

        for i in range(len(self.Weights)):
            new_input = self.Weights[i].dot(self.Layers[i])
            np.add(new_input, self.Biases[i])
            for j in range(len(new_input)):
                new_input[j] = activate(new_input[j]) + (1 if i == len(self.Weights)-1 else 0)
            self.Layers.append(new_input)

        return self.Layers[-1]

    def back_propagation_error(self, expected, derivative, c):
        self.Errors.append(c(expected, self.Layers[-1]))

        for i in range(len(self.Layers)-2, 0, -1):
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
        return y * (1-y)

    @staticmethod
    def subtract_cost_function(expected, result):
        return np.subtract(expected, result)


if __name__ == "__main__":

    train_data_filename = "classification\\data.simple.train.1000.csv"
    test_data_filename = "classification\\data.simple.test.1000.csv"

    learning_factor = 1
    input_size = 2
    output_size = 1

    data = load_csv(train_data_filename)

    network = NeuralNetwork(3, [input_size, 2, output_size])

    for ind in range(1, len(data)):
        result = network.forward_propagation(data[ind][0:input_size], NeuralNetwork.sigmoid)
        network.back_propagation_error(data[ind][input_size:len(data)], NeuralNetwork.sigmoid_derivative,
                                       NeuralNetwork.subtract_cost_function)
        network.update_weights(learning_factor)

    test_data = load_csv(test_data_filename)

    for ind in range(1, 10):
        result = network.forward_propagation(test_data[ind][0:input_size], NeuralNetwork.sigmoid)
        print("Computed = {} Actual = {} \n".format(result, test_data[ind][input_size:len(data)]))





