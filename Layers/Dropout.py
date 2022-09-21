import numpy as np
from Layers import Base


class Dropout(Base.BaseLayer):
    def __init__(self, probability):
        # Drop out with prob. 1-p
        # Divide byy p after dropping in training
        # In testing no div by p neccessary
        super().__init__()
        self.trainable = False
        self.p = probability

    def forward(self, input_tensor):
        self._out_shape = input_tensor.shape
        # Make mask. Drop with probability 1 - probability
        self.mask = np.random.uniform(low=0., high=1., size=self._out_shape) > (1-self.p)

        if self.testing_phase:
            self.output = input_tensor

        else:
            self.output = self.mask * input_tensor * (1 / self.p)

        return self.output

    def backward(self, error_tensor):
        return self.mask * error_tensor * (1/self.p)