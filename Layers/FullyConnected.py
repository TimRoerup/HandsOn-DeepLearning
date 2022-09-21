from Layers import Base
from Optimization import Optimizers
import numpy as np


class FullyConnected(Base.BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self._optimizer = None
        # Init weight matrix. Output size rows and input size + 1 columns (due to bias)
        # Question: Das input_sie already include the [x1, x2, ..., 1] 1 for the bias?
        # Weights has input size rows, output size columns (sticking to lecture)
        # self.weights = np.random.rand(self.output_size, self.input_size+1)
        self.weights = np.random.rand(self.input_size + 1, self.output_size)
        self._gradient_weights = None
        self.recent_input = None

    def initialize(self, weights_initializer, bias_initializer):
        # To make rest of the code more compact
        # Tuple: (fan_in, fan_out)
        fan_in_out = (self.input_size, self.output_size)
        
        # Bias is a single row
        self.bias = bias_initializer.initialize((1, self.output_size), *fan_in_out)
        # Weights matrix has bias added to it in via append
        self.weights = np.append(weights_initializer.initialize((self.input_size, self.output_size), *fan_in_out), self.bias, axis=0)


    def forward(self, input_tensor):
        # Store input for backpropagation
        #self.recent_input = np.insert(input_tensor.T, len(input_tensor.T), 1, axis=0)
        ones = np.ones((1, input_tensor.shape[0])).T
        self.recent_input = np.concatenate((input_tensor, ones), axis=1)
        # Input tensor is a matrix with input_size columns and batch_size rows
        #self.recent_output = (self.weights @ self.recent_input).T
        self.recent_output = self.recent_input @ self.weights

        return self.recent_output

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    def backward(self, error_tensor):
        # Need to compute gradient w.r.t. input X. Moreover, gradient w.r.t. weights
        # Gradient w.r.t. to input: E_n-1 = W.T @ E_n
        grad_X = error_tensor @ self.weights.T
        #grad_X = error_tensor @ self.weights[:,:-1]
        #grad_W = error_tensor.T @ self.recent_input.T
        grad_W = self.recent_input.T @ error_tensor
        self._gradient_weights = grad_W
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, grad_W)
        else:
            print("No optimizer specified")
        return grad_X[:,:-1]