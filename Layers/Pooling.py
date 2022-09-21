from distutils.log import error
import numpy as np
from .Base import BaseLayer


class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        """Pooling layer for convolutional neural networks.

        :param stride_shape: Stride value for pooling
        :param pooling_shape: Kernel shape for pooling
        """
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape

    def __singlePool(self, input_matrix):
        """Does a single pooling operation for a given input matrix

        :param input_matrix: The input matrix to be pooled
        :return: Pooled matrix
        """
        assert tuple(input_matrix.shape) == tuple(self.pooling_shape), f"Input matrix {input_matrix.shape} is not the same shape as pooling shape {self.pooling_shape}"
        
        index = np.unravel_index(np.argmax(np.ravel(input_matrix)), self.pooling_shape)
        highest = input_matrix[index]
        
        return index, highest
    

    def __2DPool(self, input_matrix):
        """ Does complete pooling over a 2D array

        :param input_matrix: 2D array
        :return: Pooled array
        """
        assert input_matrix.ndim == 2, "Input array is not 2D"
        
        height, width = input_matrix.shape
        
        output_shape = ((height - self.pooling_shape[0]) // self.stride_shape[0] + 1,
                        (width - self.pooling_shape[1]) // self.stride_shape[1] + 1)
        
        pooled_matrix = np.zeros(output_shape)
        # We need to store the 2D coordinates for every element in the kernal
        indexes = np.zeros((*output_shape, 2))
        
        p_i, p_j = 0, 0
        
        # I wonder if  there is a better way to do it other than for loops?
        for i in range(0, height, self.stride_shape[0]):
            for j in range(0, width, self.stride_shape[1]):
                if (i+self.pooling_shape[0] <= height) and (j+self.pooling_shape[1] <= width):
                    to_pool = input_matrix[i:i+self.pooling_shape[0], j:j+self.pooling_shape[1]]
                    index, highest = self.__singlePool(to_pool)
                
                    # Store the index and the highest value
                    indexes[p_i][p_j] = index
                    pooled_matrix[p_i][p_j] = highest
                p_j += 1
            p_i += 1
            p_j = 0
            
        return indexes, pooled_matrix
    
    def forward(self, input_tensor):
        """Forward pass of pooling layer

        :param input_tensor: Output tensor from the lower layer
        :return: The input tensor for the next layer.
        """

        # input_tensor shape: batch, channel, x, y
        # For Example: (4,4) matrix with (2,2) pool and (1,1) stride -> (4-2)//1 + 1 = 3
        output_shape = (input_tensor.shape[0],
                        input_tensor.shape[1],
                        (input_tensor.shape[2] - self.pooling_shape[0]) // self.stride_shape[0] + 1,
                        (input_tensor.shape[3] - self.pooling_shape[1]) // self.stride_shape[1] + 1)
        
        # This is the final matrix
        pooled_matrix = np.zeros(output_shape)
        # Indices are 1 where there is a maximum. Needed for backpropagation.
        self.max_indexes = np.zeros((*output_shape, 2))
        self.backward_shape = input_tensor.shape
        
        for b, batch in enumerate(input_tensor):
            for c, channel in enumerate(batch):
                self.max_indexes[b][c], pooled_matrix[b][c] = self.__2DPool(channel)
        
        return pooled_matrix
    
    def backward(self, error_tensor):
        """ Backward pass of pooling layer

        :param error_tensor: Gradient tensor from the upper layer.
        :return: Gradient w.r.t. input tensor that serves as input to the lower layer during backpropagation
        """
        output_shape = self.backward_shape
        
        output_matrix = np.zeros(output_shape)
        
        for b in range(error_tensor.shape[0]):
            for c in range(error_tensor.shape[1]):
                for i in range(error_tensor.shape[2]):
                    for j in range(error_tensor.shape[3]):
                        max_index = self.max_indexes[b][c][i][j]
                        loc_i = i * self.stride_shape[0] + int(max_index[0])
                        loc_j = j * self.stride_shape[1] + int(max_index[1])
                        output_matrix[b][c][loc_i][loc_j] += error_tensor[b][c][i][j]
        return output_matrix
