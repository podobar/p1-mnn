import numpy as np


class CostFunctions:

    @staticmethod
    def MSE_gradient(expected, result):
        return np.subtract(expected, result)

    @staticmethod
    def cross_entropy_gradient(expected, result):
        return np.divide(np.subtract(expected, result), [(1-a)*a for a in result])




