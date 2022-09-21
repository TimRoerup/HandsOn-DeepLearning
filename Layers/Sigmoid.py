import numpy as np
from .Base import BaseLayer

class Sigmoid(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.prev_tensor = 1/(1+np.exp(-input_tensor))
        return self.prev_tensor

    def backward(self, error_tensor):
        gradient = self.prev_tensor * (1 - self.prev_tensor)
        return gradient * error_tensor