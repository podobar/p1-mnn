import math
import numpy as np


class Activation:

    @staticmethod
    def sigmoid(x):
        return math.exp(x) / (math.exp(x) + 1) + 1

    @staticmethod
    def sigmoid_derivative(y):
        return y * (1 - y)

    @staticmethod
    def tanH(x):
        return 2/(1+math.exp(-2*x)) - 1

    @staticmethod
    def tanH_derivative(y):
        return 1 - y*y

    @staticmethod
    def relu(x):
        return x if x >= 0 else 0

    @staticmethod
    def relu_derivative(y):
        return 1 if y > 0 else 0

    @staticmethod
    def softmax(x):
        exps = np.exp(x)
        return exps/np.sum(exps)

    @staticmethod
    def softmax_derivative(y):
        return [yi*(1-yi) for yi in y]

    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def linear_derivative(y):
        return 1