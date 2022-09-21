import numpy as np


class CrossEntropyLoss():
    def __init__(self):
        self.eps = np.finfo(float).eps

    def forward(self, prediction_tensor, label_tensor):
        # Remember the prediction_tensor
        self.prev_tensor = prediction_tensor

        # Get the indexes from where label_tensor = 1
        indexes = (label_tensor == 1).nonzero()

        # Access the same indexes in prediction_tensor prediction_tensor[indexes]
        # Apply Cross Entropy Formula
        sum_inside = np.reshape(-np.log(prediction_tensor[indexes] + self.eps), (-1,1))
        return np.sum(sum_inside)

    def backward(self, label_tensor):
        return - label_tensor / self.prev_tensor