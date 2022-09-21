import numpy as np
from Layers import Base

class BatchNormalization(Base.BaseLayer):
    def __init__(self, channels):
        """ Batch normalization layer. Works for fully connected layer and convolution layer.
        :param channels: Channels of input tensor in vector and image case
        """

        super().__init__()
        self.channels = channels
        self.trainable = True
        self.testing_phase = False
        self.optimizer = None

        self.weights = np.ones((self.channels))
        self.bias = np.zeros((self.channels))
        self.fc = False
        self._shape_tensor = None
        self.mean = None
        self.sigma = None
        self.a = 0.8
        self.gradient_weights = None
        self.gradient_bias = None
        self._gradient_input = None
        self.input_tensor = None

    def compute_bn_gradients(self, error_tensor, input_tensor, weights, mean, var, eps=np.finfo(float).eps):
        """ Computation of the gradient w.r.t the input for the batch_normalization layer.

        :param error_tensor: Gradient tensor from the upper layer
        :param input_tensor: Output tensor from the lower layer
        :param weights: Weights for the batch normalization layer
        :param mean: Mean of the batch (calculated during training via sliding window approach)
        :param var: Variance of the batch (calculated during training via sliding window approach)
        :param eps: Vanishing small float to avoid dividing by zero.
        :return: Gradient w.r.t the input tensor
        """

        if eps > 1e-10:
            raise ArithmeticError("Eps must be lower than 1e-10. Your eps values %s" % (str(eps)))

        norm_mean = input_tensor - mean
        var_eps = var + eps

        gamma_err = error_tensor * weights
        inv_batch = 1. / error_tensor.shape[0]

        grad_var = np.sum(norm_mean * gamma_err * -0.5 * (var_eps ** (-3 / 2)), keepdims=True, axis=0)

        sqrt_var = np.sqrt(var_eps)
        first = gamma_err * 1. / sqrt_var

        grad_mu_two = (grad_var * np.sum(-2. * norm_mean, keepdims=True, axis=0)) * inv_batch
        grad_mu_one = np.sum(gamma_err * -1. / sqrt_var, keepdims=True, axis=0)

        second = grad_var * (2. * norm_mean) * inv_batch
        grad_mu = grad_mu_two + grad_mu_one

        return first + second + inv_batch * grad_mu

    def forward(self, input_tensor):
        """ Batch normalization forward pass.

        :param input_tensor: Output tensor from the lower layer
        :return: The input tensor for the next layer.
        """

        # Check whether Convolution Layer
        if input_tensor.ndim == 4:
            self.fc = False
            input_tensor = self.reformat(input_tensor)
        # FC layer case
        else:
            self.fc = True
            input_tensor = input_tensor

        self.input_tensor = input_tensor
        e = np.finfo(float).eps

        # Train time: replace mean_b and sigma_b by mean and sigma of training data (same dimension as input)
        if not self.testing_phase:
            # Normalize X. beta, gamma, sigma²_B, mü_B are vectors with len = self.channels
            mean_b = np.mean(input_tensor, axis=0)
            sigma_b = np.std(input_tensor, axis=0)**2
            # Init data mean and variance if not yet done
            if self.mean is None:
                self.mean = mean_b
                self.sigma = sigma_b

            self.mean = self.a * self.mean + (1-self.a) * mean_b
            self.sigma = self.a * self.sigma + (1-self.a) * sigma_b

            self.X_tilde = (input_tensor - mean_b) / ((sigma_b + e) ** 0.5)
            self.Y_head = self.weights * self.X_tilde + self.bias

        # Test time: Update mean and sigma
        if self.testing_phase:
            self.X_tilde = (input_tensor - self.mean) / ((self.sigma + e) ** 0.5)
            self.Y_head = self.weights * self.X_tilde + self.bias

        # Reshape with the reversed procedure for Convolution layer
        self.Y_head = self.Y_head if self.fc else self.reformat(self.Y_head)

        return self.Y_head


    def backward(self, error_tensor):
        """Batch normalization backward pass

        :param error_tensor: Gradient tensor from the upper layer.
        :return: Gradient w.r.t. input tensor that serves as input to the lower layer during backpropagation
        """
        self.gradient_weights = np.zeros_like(self.weights)
        self.gradient_bias = np.zeros_like(self.bias)

        if error_tensor.ndim == 4:
            # We have a CV layer -> Reshape input
            self.fc = False
            error_tensor = self.reformat(error_tensor)
        # FC layer case
        else:
            self.fc = True

        for b in range(error_tensor.shape[0]):
            self.gradient_weights += error_tensor[b] * self.X_tilde[b]

        self.gradient_bias = np.sum(error_tensor, axis=0)

        self._gradient_input = self.compute_bn_gradients(error_tensor, self.input_tensor, self.weights, self.mean, self.sigma)
        self._gradient_input = self._gradient_input if self.fc else self.reformat(self._gradient_input)

        # Update weights & bias
        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = self.optimizer.calculate_update(self.bias, self.gradient_bias)

        return self._gradient_input

    def initialize(self, weights_initializer, bias_initializer):
        """Initializes weights and bias for the batch normalization layer

        :param weights_initializer: A weight initializer from Initializers.py
        :param bias_initializer: A bias initializer from Initializers.py
        :return: Weights and biases
        """
        # Gets initializers but ignores them intentionally
        self.weights = np.ones((self.channels))
        self.bias = np.zeros((self.channels))

    def reformat(self, tensor):
        """Reshapes input tensor to be able to deal with inputs from a convolutional layer and fully connected layer as
        well.

        :param tensor: Tensor to be reshaped
        :return: Reshaped tensor
        """
        if tensor.ndim == 4:
            # We have a CV layer -> Reshape input
            self.fc = False
            self._shape_tensor = tensor.shape
            # B x H x M x N -> B x H x M*N
            tensor = tensor.reshape((*tensor.shape[0:2], -1))
            # B x H x M*N -> B x M*N x H
            tensor = np.swapaxes(tensor, 1, 2)
            # B x M*N x H -> B*M*N x H
            tensor = tensor.reshape((-1, tensor.shape[-1]))

            return tensor
        # Procedure from above reversed
        else:
            B, H, M, N = self._shape_tensor
            tensor = tensor.reshape((B, M*N, H))
            tensor = np.swapaxes(tensor, 1, 2)
            tensor = tensor.reshape(B, H, M, N)

            return tensor
