import numpy as np

class MeanSquaredError:
    """Mean Squared Error loss for regression tasks."""
    def __init__(self):
        self.predictions = None
        self.targets = None

    def forward(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets
        return np.mean((predictions - targets) ** 2)

    def backward(self):
        batch_size = self.predictions.shape[0]
        return 2 * (self.predictions - self.targets) / batch_size


class CategoricalCrossEntropy:
    """
    Combined Softmax activation and Categorical Cross-Entropy loss.
    Combining them ensures numerical stability during backpropagation.
    """
    def __init__(self):
        self.probs = None
        self.targets = None

    def forward(self, logits, targets):
        """
        logits shape: (batch_size, num_classes)
        targets shape: (batch_size, num_classes) - one-hot encoded
        """
        # Numerical stability shift
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        self.probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        self.targets = targets
        
        batch_size = logits.shape[0]
        # Prevent log(0) with epsilon
        eps = 1e-15
        clipped_probs = np.clip(self.probs, eps, 1 - eps)
        loss = -np.sum(targets * np.log(clipped_probs)) / batch_size
        return loss

    def backward(self):
        """
        Gradient of loss with respect to logits: (P - Y) / N
        """
        batch_size = self.targets.shape[0]
        return (self.probs - self.targets) / batch_size