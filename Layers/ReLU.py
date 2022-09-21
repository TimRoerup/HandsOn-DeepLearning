import numpy as np
import sys
from .Base import BaseLayer


class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()
        self.prev_input = None

    def forward(self, input_tensor):
        # Store input for backpropagation
        self.prev_input = np.copy(input_tensor)

        # Apply ReLU function on input_tensor
        input_tensor[input_tensor < 0] = 0
        return np.copy(input_tensor)

    def backward(self, error_tensor):
        # ReLU differential: 0 if x<0, 1 otherwise
        # Similar to forward-pass, but uses differentiated ReLU
        error_tensor[self.prev_input < 0] = 0

        return np.copy(error_tensor)