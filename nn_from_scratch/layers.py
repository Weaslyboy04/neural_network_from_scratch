import numpy as np

class Layer:
    """Base class for all neural network layers."""
    def __init__(self):
        self.params = {}
        self.grads = {}

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError


class Dense(Layer):
    """Fully connected linear layer: Z = XW + b"""
    def __init__(self, in_features, out_features, init_method='he'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight initialization
        if init_method == 'he':
            scale = np.sqrt(2.0 / in_features)
        elif init_method == 'xavier':
            scale = np.sqrt(1.0 / in_features)
        else:
            scale = 0.01
            
        self.params['W'] = np.random.randn(in_features, out_features) * scale
        self.params['b'] = np.zeros((1, out_features))
        
        self.grads['W'] = np.zeros_like(self.params['W'])
        self.grads['b'] = np.zeros_like(self.params['b'])
        self.inputs = None

    def forward(self, inputs):
        """
        inputs shape: (batch_size, in_features)
        returns shape: (batch_size, out_features)
        """
        self.inputs = inputs
        return np.dot(inputs, self.params['W']) + self.params['b']

    def backward(self, grad_output):
        """
        grad_output shape: (batch_size, out_features)
        returns grad_input shape: (batch_size, in_features)
        """
        self.grads['W'] = np.dot(self.inputs.T, grad_output)
        self.grads['b'] = np.sum(grad_output, axis=0, keepdims=True)
        grad_input = np.dot(grad_output, self.params['W'].T)
        return grad_input