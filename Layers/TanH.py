import numpy as np
from .Base import BaseLayer

class TanH(BaseLayer):
    """ Tangents Hyperbolicus activation function/layer  """
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        """ Forward pass of Tangents Hyperbolicus layer. Basically applies ReLU function.

        :param input_tensor: Output tensor from the lower layer
        :return: The input tensor for the next layer
        """
        self.prev_tensor = np.tanh(input_tensor)
        return self.prev_tensor

    def backward(self, error_tensor):
        """ Backward pass of Tangents Hyperbolicus layer

        :param error_tensor: Gradient tensor from the upper layer
        :return: Gradient w.r.t. input tensor that serves as input to the lower layer during backpropagation
        """
        gradient = 1 - self.prev_tensor**2
        return gradient * error_tensor