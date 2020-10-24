import numpy as np


class CostFunctions:

    @staticmethod
    def MSE_gradient(expected, result):
        return np.subtract(expected, result)

    @staticmethod
    def cross_entropy_gradient(expected, result):
        return np.divide(np.subtract(expected, result), [(1-a)*a for a in result])

    @staticmethod
    def MSE(n_expected, n_result):
        dtab = list()
        for i in range(len(n_expected)):
            dtab.append(np.power(np.subtract(n_expected[i], n_result[i]), 2).sum())
        return np.mean(dtab)





