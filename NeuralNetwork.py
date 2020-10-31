import numpy as np

from Visualization import Visualization

class NeuralNetwork:

    Layers = list()
    Weights = list()
    Biases = list()
    Errors = list()

    def __init__(self, sizes):

        for i in range(len(sizes)-1):
            weights = np.random.rand(sizes[i+1], sizes[i])
            self.Weights.append(weights)

            bias = np.random.rand(sizes[i+1])
            self.Biases.append(bias)

    def forward_propagation(self, input, inner_activate, out_activate):
        self.Layers.clear()
        self.Layers.append(input)

        for i in range(len(self.Weights)):
            new_input = self.Weights[i].dot(self.Layers[i])
            np.add(new_input, self.Biases[i])

            f = out_activate if i == len(self.Weights) - 1 else inner_activate
            new_input = f(new_input)

            self.Layers.append(new_input)

        return self.Layers[-1]

    def back_propagation_error(self, expected, inner_derivative, out_derivative, cost_gradient):
        errors = cost_gradient(expected, self.Layers[-1])
        errors = np.multiply(errors, out_derivative(self.Layers[-1]))
        self.Errors.append(errors)

        for i in range(len(self.Layers)-2, 0, -1):
            errors = list()
            for j in range(len(self.Layers[i])):
                error = 0
                for k in range(len(self.Layers[i+1])):
                    error += self.Weights[i][k][j] * self.Errors[-1][k]
                errors.append(error)

            errors = np.multiply(errors, inner_derivative(self.Layers[i]))
            self.Errors.append(errors)

        self.Errors.reverse()
        #Visualization.write_out_neuron_errors(self)

    def update_weights(self, factor):
        for i in range(len(self.Weights)):
            for j in range(len(self.Layers[i+1])):
                for k in range(len(self.Layers[i])):
                    self.Weights[i][j][k] += factor * self.Errors[i][j] * self.Layers[i][k]
                self.Biases[i][j] += factor * self.Errors[i][j]
        self.Errors.clear()
        #Visualization.write_out_neural_network_weights(self)

    def learn(self, input_size, train_set, val_set, inner_activate, out_activate, inner_derivative, out_derivative,
              cost_gradient, learn_factor, iterations):

        bestW = list()
        bestB = list()
        minloss = -1

        for i in range(iterations):
            data = train_set[i]
            self.forward_propagation(data[0:input_size], inner_activate, out_activate)
            self.back_propagation_error(data[input_size:len(data)], inner_derivative, out_derivative, cost_gradient)
            self.update_weights(learn_factor)

            actual = list()
            predicted = list()
            for val_data in val_set:
                actual.append(val_data[input_size:len(data)])
                predicted.append(self.forward_propagation(val_data[0:input_size], inner_activate, out_activate))

        #     loss = loss_function(actual, predicted)
        #     if (minloss == -1) | (minloss > loss):
        #         minloss = loss
        #         bestW = self.Weights
        #         bestB = self.Biases
        #
        # self.Weights = bestW
        # self.Biases = bestB








