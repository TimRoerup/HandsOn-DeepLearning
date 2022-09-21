import numpy as np
from .Base import BaseLayer

class TanH(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.prev_tensor = np.tanh(input_tensor)
        return self.prev_tensor

    def backward(self, error_tensor):
        gradient = 1 - self.prev_tensor**2
        return gradient * error_tensor