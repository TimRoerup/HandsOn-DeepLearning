import numpy as np

class L2_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        # What exactly calculate here?
        return self.alpha * weights

    def norm(self, weights):
        return self.alpha * np.linalg.norm(weights)**2

class L1_Regularizer:
    # Sparser weights
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        return self.alpha * np.sign(weights)

    def norm(self, weights):
        # L1 norm: Max row sum of |absolut| values
        return self.alpha * np.sum(np.linalg.norm(weights, ord=1, axis=1))