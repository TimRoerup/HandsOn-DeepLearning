import numpy as np
import copy
from Layers import *
from Optimization import *


class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self._phase = None

    def forward(self):
        # Get the input and label tensors from data_layer
        self.input_tensor, self.label_tensor = self.data_layer.next()

        self.reg_loss = 0
        # Pass input through each layer (excluding loss)
        for layer in self.layers:
            self.input_tensor = layer.forward(self.input_tensor)
            if layer.trainable and layer.optimizer.regularizer:
                self.reg_loss += layer.optimizer.regularizer.norm(layer.weights)

        # Get the loss values from our predictions
        loss = self.loss_layer.forward(self.input_tensor, self.label_tensor)
        return loss if not self.reg_loss else loss + self.reg_loss

    def backward(self):
        # Goes through loss first, and then the rest of the network backwards
        self.label_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in self.layers[::-1]:
            self.label_tensor = layer.backward(self.label_tensor)

    def append_layer(self, layer):
        if layer.trainable:
            # Makes a deep copy of layers optimizer 
            # And sets it to this layer
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)

        self.layers.append(layer)

    # Trains the network for # of iterations specified.
    def train(self, iterations):
        # Set phase to train
        self.phase = "train"
        # Do a forward pass, store the loss
        # Apply backpropagation
        # Repeat iteration times.
        for i in range(iterations):
            loss_i = self.forward()
            self.loss.append(loss_i)
            self.backward()

    def test(self, input_tensor):
        # Set phase to test
        self.phase = "test"

        for layer in self.layers:
            layer.testing_phase = True
            input_tensor = layer.forward(input_tensor)
        return np.copy(input_tensor)

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, value):
        self._phase = value