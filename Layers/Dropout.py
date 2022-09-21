import numpy as np
from Layers import Base


class Dropout(Base.BaseLayer):
    """ Dropout layer - Randomly drops some nodes/connections.

    :param probability: Value between 0 and 1 determine the probability to keep connections. 1->Keep all ; 0->Drop all
    """
    def __init__(self, probability):
        super().__init__()
        self.trainable = False
        self.p = probability
        self._out_shape = None
        self.mask = None
        self.output = None

    def forward(self, input_tensor):
        """Dropout layer forward pass.

        :param input_tensor: The output tensor of the previous layer.
        :return: The input tensor for the next layer.
        """
        self._out_shape = input_tensor.shape
        self.mask = np.random.uniform(low=0., high=1., size=self._out_shape) > (1-self.p)

        # In testing, the dropout layer is turned off.
        if self.testing_phase:
            self.output = input_tensor
        # In training, the dropout layer is active.
        else:
            self.output = self.mask * input_tensor * (1 / self.p)

        return self.output

    def backward(self, error_tensor):
        """Dropout layer backward pass

        :param error_tensor: Gradient tensor from the upper layer.
        :return: Gradient tensor
        """
        return self.mask * error_tensor * (1/self.p)