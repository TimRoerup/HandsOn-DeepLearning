import numpy as np


class CrossEntropyLoss():
    """Cross entropy loss function/layer"""
    def __init__(self):
        self.eps = np.finfo(float).eps

    def forward(self, prediction_tensor, label_tensor):
        """Forward pass of CrossEntropy layer

        :param prediction_tensor: Output tensor. CE layer is very likely the last layer so the output can be
        interpreted as predictions.
        :param label_tensor: The actual labels
        :return: Loss vector/array
        """
        # Remember the prediction_tensor
        self.prev_tensor = prediction_tensor

        # Get the indexes from where label_tensor = 1
        indexes = (label_tensor == 1).nonzero()

        # Access the same indexes in prediction_tensor prediction_tensor[indexes]
        # Apply Cross Entropy Formula
        sum_inside = np.reshape(-np.log(prediction_tensor[indexes] + self.eps), (-1,1))
        return np.sum(sum_inside)

    def backward(self, label_tensor):
        """Backward pass of CrossEntropy layer

        :param label_tensor: The actual labels
        :return: Gradient w.r.t. input tensor that serves as input to the lower layer during backpropagation
        """
        return - label_tensor / self.prev_tensor