import numpy as np


class Scaler:

    Mean = 0
    Std_dev = 1
    Range = 0
    Min = 0

    def fit(self, vector):
        self.Mean = np.mean(vector)
        self.Std_dev = np.std(vector)
        self.Range = np.max(vector) - np.min(vector)
        self.Min = np.min(vector)

    def standardize(self, vector):
        return np.divide(np.subtract(vector, self.Mean), self.Std_dev)

    def unstandardize(self, vector):
        return np.add(np.multiply(vector, self.Std_dev), self.Mean)

    def normalize(self, vector):
        return np.divide(np.subtract(vector, self.Min), self.Range)

    def unnormalize(self, vector):
        return np.multiply(np.add(vector, np.min(vector)), self.Range)

    @staticmethod
    def identity(vector):
        return vector