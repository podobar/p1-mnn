import numpy as np


class CostFunctions:

    @staticmethod
    def MSE_gradient(expected, result):
        rarray = np.subtract(expected, result)
        return rarray

    @staticmethod
    def kullback_gradient(expected, result):
        return np.divide(expected, result)

    @staticmethod
    def cross_entropy_gradient(expected, result):
        return np.divide(np.subtract(expected, result), [(1-a)*a for a in result])

    @staticmethod
    def MSE(n_expected, n_result):
        dtab = list()
        for i in range(len(n_expected)):
            dtab.append(np.mean(np.power(np.subtract(n_expected[i], n_result[i]), 2)))
        return np.mean(dtab)

    @staticmethod
    def Kullback(n_expected, n_result):
        dtab = list()
        for i in range(len(n_expected)):
            dtab.append(np.multiply(np.log(np.divide(n_expected[i], n_result[i])), n_expected[i]).sum())
        return np.sum(dtab)





