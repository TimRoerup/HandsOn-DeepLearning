import numpy as np
from Layers import Base


class Flatten(Base.BaseLayer):
    """Flatten layer that flattens the input tensor.
    """
    def __init__(self):
        super().__init__()
        self.shape_in = None

    def forward(self, input_tensor):
        """Flattens each multidimensional sample from the batch of the input tensor.

        :param input_tensor: Input tensor to be flattened.
        :return: Flattened tensor consisting of 1D batch-samples.
        """
        self.shape_in = np.shape(input_tensor)
        batch_size = self.shape_in[0]
        return input_tensor.reshape(batch_size, -1)

    def backward(self, error_tensor):
        """Takes a flattened tensor and reshapes it to its original shape.

        :param error_tensor: Flattened gradient tensor from upper layer.
        :return: Original shaped gradient tensor.
        """
        return error_tensor.reshape(self.shape_in)