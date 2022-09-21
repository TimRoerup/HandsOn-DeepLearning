import numpy as np

class Optimizer:
    def __init__(self):
        self.regularizer = None
        self.learning_rate = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer


class Sgd(Optimizer):
    def __init__(self, learning_rate: float):
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer:
            return weight_tensor - self.learning_rate * (self.regularizer.calculate_gradient(weight_tensor) + gradient_tensor)
        else:
            return weight_tensor  - self.learning_rate * gradient_tensor

class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        # Previous moment is initialized as zero (I think this makes sense?)
        # Caching of layer specific weights are done in Network side, I believe? (Where the optimizer is initiated)
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.prev_moment = 0
    
    def calculate_update(self, weight_tensor, gradient_tensor):
        # Calculate the new moment, store it for later use, return the updated weights
        moment = self.momentum_rate * self.prev_moment - self.learning_rate * gradient_tensor
        self.prev_moment = moment
        if self.regularizer:
            return weight_tensor - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor) + moment
        else:
            return weight_tensor + moment

class Adam(Optimizer):
    def __init__(self, learning_rate, mu, rho):
        super().__init__()

        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        
        # The previous iteration values, initialized as 0
        self.prev_moment = 0
        self.prev_sec_moment = 0
        
        # Iteration number for bias correction
        self.iter_no = 1
        
        # Smallest representable number, avoids division by 0
        self.eps = np.finfo(float).eps
    
    def calculate_update(self, weight_tensor, gradient_tensor):
        # Calculate moment v^k and update the previous one
        moment = self.mu * self.prev_moment + (1-self.mu) * gradient_tensor
        self.prev_moment = moment
        
        # Calculate second moment r^k and update the previous one
        sec_moment = self.rho * self.prev_sec_moment + (1-self.rho) * np.square(gradient_tensor)
        self.prev_sec_moment = sec_moment
        
        # Bias Correction, and increasing the iteration number
        cor_moment = moment / (1 - self.mu**self.iter_no)
        cor_sec_moment = sec_moment / (1 - self.rho**self.iter_no)
        self.iter_no += 1
        
        # Calculate the updated weights
        if self.regularizer:
            return weight_tensor - self.learning_rate * \
                   (self.regularizer.calculate_gradient(weight_tensor) +
                    cor_moment / (np.sqrt(cor_sec_moment) + self.eps))
        else:
            return weight_tensor - self.learning_rate * cor_moment / (np.sqrt(cor_sec_moment) + self.eps)