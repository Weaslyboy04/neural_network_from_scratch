import numpy as np
from .layers import Layer

class ReLU(Layer):
    """Rectified Linear Unit activation function."""
    def __init__(self):
        super().__init__()
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, grad_output):
        grad_input = grad_output.copy()
        grad_input[self.inputs <= 0] = 0
        return grad_input


class Sigmoid(Layer):
    """Sigmoid activation function."""
    def __init__(self):
        super().__init__()
        self.outputs = None

    def forward(self, inputs):
        # Clip inputs to prevent overflow
        inputs_clipped = np.clip(inputs, -500, 500)
        self.outputs = 1.0 / (1.0 + np.exp(-inputs_clipped))
        return self.outputs

    def backward(self, grad_output):
        return grad_output * self.outputs * (1.0 - self.outputs)