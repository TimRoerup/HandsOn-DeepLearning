import numpy as np
from Layers import Base


class SoftMax(Base.BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        
        # Max shift: x_i' <- x_i - max(x)
        # Get maxima of every batch entry (row of input_tensor)
        maxima = np.reshape(np.amax(input_tensor, axis=1), (-1, 1))
        maxima = np.tile(maxima, (1, input_tensor.shape[1]))
        # Adapt input tensor by subtracting max
        input_tensor = input_tensor - maxima
    
        # Softmax
        # Calculate exponential sum over rows of input_tensor
        exp_sums = np.exp(input_tensor)
        total_exp_sums =  np.reshape(np.sum(exp_sums, axis=1), (-1,1))
        
        # Apply softmax function. Each input_tensor entry gets divided by the corresponding scaler value
        self.y = exp_sums / np.tile(total_exp_sums, (1, input_tensor.shape[1]))

        return self.y

    def backward(self, error_tensor):
        sums = np.sum(error_tensor * self.y, axis=1).reshape(-1, 1)
        sums = np.tile(sums, (1, error_tensor.shape[1]))

        return self.y * (error_tensor - sums)
