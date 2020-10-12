import numpy as np
import math


class NeuralNetwork:

    Layers = list()
    Weights = list()
    Biases = list()

    def __init__(self, n:int, sizes):

        for i in range(len(sizes)-1):
            weights = np.random.rand(sizes[i+1], sizes[i])
            self.Weights.append(weights)

            bias = np.random.rand(sizes[i+1])
            self.Biases.append(bias)

    def forward_propagation(self, input: np.array, activate):
        self.Layers.append(input)

        for i in range(len(self.Weights)):
            new_input = self.Weights[i].dot(self.Layers[i])
            np.add(new_input, self.Biases[i])
            for j in range(len(new_input)):
                new_input[j] = activate(new_input[j])
            self.Layers.append(new_input)

        return self.Layers[-1]

    @staticmethod
    def sigmoid(x):
        return math.exp(x)/(math.exp(x) + 1)


if __name__ == "__main__":

    network = NeuralNetwork(3, [2, 3, 1])
    result = network.forward_propagation([0, 1], NeuralNetwork.sigmoid)
    print(result)




