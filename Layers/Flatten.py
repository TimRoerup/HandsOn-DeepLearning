import numpy as np
from Layers import Base


class Flatten(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        self.shape_in = None

    def forward(self, input_tensor):
        self.shape_in = np.shape(input_tensor)
        batch_size = self.shape_in[0]
        return input_tensor.reshape(batch_size, -1)

    def backward(self, error_tensor):
        return error_tensor.reshape(self.shape_in)