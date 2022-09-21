import numpy as np
from .Base import BaseLayer

class Sigmoid(BaseLayer):
    """ Sigmoid activation function/layer """
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        """ Forward pass of Sigmoid layer. Basically applies sigmoid function.

        :param input_tensor: Output tensor from the lower layer
        :return: The input tensor for the next layer
        """
        self.prev_tensor = 1/(1+np.exp(-input_tensor))
        return self.prev_tensor

    def backward(self, error_tensor):
        """ Backward pass of Sigmoid layer

        :param error_tensor: Gradient tensor from the upper layer
        :return: Gradient w.r.t. input tensor that serves as input to the lower layer during backpropagation
        """
        gradient = self.prev_tensor * (1 - self.prev_tensor)
        return gradient * error_tensor