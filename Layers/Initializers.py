import numpy as np

# To be done: Consider if fully connected layer or convolutional layer

class Constant:
    def __init__(self, c=0.1):
        self.constant = c

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.full(weights_shape, self.constant).reshape(weights_shape)



class UniformRandom:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        #return np.random.rand(fan_in, fan_out).reshape(weights_shape)
        return np.random.uniform(size=weights_shape)


class Xavier:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.normal(loc=0, scale=(2 / (fan_in + fan_out)) ** 0.5, size=weights_shape)


class He:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.normal(loc=0, scale=(2 / fan_in) ** 0.5, size=weights_shape)