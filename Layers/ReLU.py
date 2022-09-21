import numpy as np
import sys
from .Base import BaseLayer


class ReLU(BaseLayer):
    def __init__(self):
        """ Rectified linear unit activation function/layer
        """
        super().__init__()
        self.prev_input = None

    def forward(self, input_tensor):
        """ Forward pass of ReLU layer. Basically applies ReLU function.

        :param input_tensor: Output tensor from the lower layer
        :return: The input tensor for the next layer
        """

        # Store input for backpropagation
        self.prev_input = np.copy(input_tensor)

        # Apply ReLU function on input_tensor
        input_tensor[input_tensor < 0] = 0

        return np.copy(input_tensor)

    def backward(self, error_tensor):
        """ Backward pass of ReLU layer

        :param error_tensor: Gradient tensor from the upper layer
        :return: Gradient w.r.t. input tensor that serves as input to the lower layer during backpropagation
        """

        # ReLU differential: 0 if x<0, 1 otherwise
        error_tensor[self.prev_input < 0] = 0

        return np.copy(error_tensor)
