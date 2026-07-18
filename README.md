# NumPy Deep Learning Library from Scratch

A clean, modular deep learning framework implemented purely in Python and NumPy—no PyTorch, TensorFlow, or autograd libraries required. 

This repository implements feedforward neural networks using an object-oriented architecture inspired by modern deep learning frameworks. It supports custom layer stacking, forward/backward propagation, mini-batch training, and advanced optimization algorithms.

## Features

- **Layers:** Fully Connected (`Dense`) with Xavier/Glorot and He normal initialization.
- **Activations:** `ReLU`, `Sigmoid`, and `Softmax`.
- **Loss Functions:** `CategoricalCrossEntropy` (numerically stabilized with Softmax) and `MeanSquaredError`.
- **Optimizers:** Standard Stochastic Gradient Descent (`SGD`) and `Adam` with momentum and RMSProp updates.
- **Model API:** A `Sequential` container that manages forward passes, backpropagation, and mini-batch training loops.

## Mathematical Background

### 1. Dense Layer Forward & Backward Pass
For a batch of input vectors $X \in \mathbb{R}^{N \times D_{in}}$, weight matrix $W \in \mathbb{R}^{D_{in} \times D_{out}}$, and bias vector $b \in \mathbb{R}^{1 \times D_{out}}$:

**Forward Pass:**
$$Z = XW + b$$

**Backward Pass:**
Given the upstream gradient $\frac{\partial L}{\partial Z} \in \mathbb{R}^{N \times D_{out}}$:
$$\frac{\partial L}{\partial W} = X^T \frac{\partial L}{\partial Z}, \quad \frac{\partial L}{\partial b} = \sum_{i=1}^{N} \left(\frac{\partial L}{\partial Z}\right)_i, \quad \frac{\partial L}{\partial X} = \frac{\partial L}{\partial Z} W^T$$

### 2. Adam Optimizer Update Rule
Adam computes adaptive learning rates for each parameter using first and second moment estimates:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

Bias-corrected estimates:
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

Parameter update:
$$\theta_{t} = \theta_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

## Installation & Usage

1. **Clone the repository and install dependencies:**
   ```bash
   git clone [https://github.com/yourusername/numpy-dl-from-scratch.git](https://github.com/yourusername/numpy-dl-from-scratch.git)
   cd numpy-dl-from-scratch
   pip install -r requirements.txt