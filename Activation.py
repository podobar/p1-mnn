import math
import numpy as np


class Activation:

    @staticmethod
    def sigmoid(x):
        x_norm = [min(xi, 10) for xi in x]
        return [math.exp(xi) / (math.exp(xi) + 1) + 1 for xi in x_norm]

    @staticmethod
    def sigmoid_derivative(y):
        return [yi * (1 - yi) for yi in y]

    @staticmethod
    def tanH(x):
        return [2/(1+math.exp(-2*xi)) - 1 for xi in x]

    @staticmethod
    def tanH_derivative(y):
        return [1 - yi*yi for yi in y]

    @staticmethod
    def relu(x):
        return [(xi if xi >= 0 else 0) for xi in x]

    @staticmethod
    def relu_derivative(y):
        return [(1 if yi > 0 else 0) for yi in y]

    @staticmethod
    def softmax(x):
        x_norm = [min(xi, 10) for xi in x]
        exps = np.exp(x_norm)
        return exps/np.sum(exps)

    @staticmethod
    def softmax_derivative(y):
        return [yi * (1 - yi) for yi in y]

    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def linear_derivative(y):
        return np.ones(len(y))