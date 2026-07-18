import numpy as np

class SGD:
    """Stochastic Gradient Descent with optional momentum."""
    def __init__(self, lr=0.01, momentum=0.0):
        self.lr = lr
        self.momentum = momentum
        self.velocities = {}

    def update(self, layers):
        for idx, layer in enumerate(layers):
            if not layer.params:
                continue
            if idx not in self.velocities:
                self.velocities[idx] = {k: np.zeros_like(v) for k, v in layer.params.items()}
                
            for param_name in layer.params:
                v = self.velocities[idx][param_name]
                g = layer.grads[param_name]
                
                v = self.momentum * v - self.lr * g
                self.velocities[idx][param_name] = v
                layer.params[param_name] += v


class Adam:
    """Adam optimizer (Adaptive Moment Estimation)."""
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, layers):
        self.t += 1
        for idx, layer in enumerate(layers):
            if not layer.params:
                continue
            if idx not in self.m:
                self.m[idx] = {k: np.zeros_like(v) for k, v in layer.params.items()}
                self.v[idx] = {k: np.zeros_like(v) for k, v in layer.params.items()}
                
            for param_name in layer.params:
                grad = layer.grads[param_name]
                
                # Update biased first moment estimate
                self.m[idx][param_name] = self.beta1 * self.m[idx][param_name] + (1 - self.beta1) * grad
                # Update biased second raw moment estimate
                self.v[idx][param_name] = self.beta2 * self.v[idx][param_name] + (1 - self.beta2) * (grad ** 2)
                
                # Compute bias-corrected estimates
                m_hat = self.m[idx][param_name] / (1 - self.beta1 ** self.t)
                v_hat = self.v[idx][param_name] / (1 - self.beta2 ** self.t)
                
                # Update parameter
                layer.params[param_name] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)