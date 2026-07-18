import numpy as np

class Sequential:
    """Linear stack of layers to build a feedforward neural network."""
    def __init__(self, layers=None):
        self.layers = layers if layers is not None else []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, inputs):
        out = inputs
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad_output):
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def fit(self, X, y, epochs, batch_size, loss_fn, optimizer, val_data=None):
        num_samples = X.shape[0]
        
        for epoch in range(1, epochs + 1):
            # Shuffle dataset every epoch
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0.0
            num_batches = int(np.ceil(num_samples / batch_size))
            
            for b in range(num_batches):
                start_idx = b * batch_size
                end_idx = min(start_idx + batch_size, num_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                predictions = self.forward(X_batch)
                loss = loss_fn.forward(predictions, y_batch)
                epoch_loss += loss * (end_idx - start_idx)
                
                # Backward pass
                grad_loss = loss_fn.backward()
                self.backward(grad_loss)
                
                # Optimizer step
                optimizer.update(self.layers)
                
            epoch_loss /= num_samples
            
            # Validation metrics
            val_info = ""
            if val_data is not None:
                X_val, y_val = val_data
                val_preds = self.forward(X_val)
                val_loss = loss_fn.forward(val_preds, y_val)
                
                # Compute accuracy if one-hot classification
                if y_val.ndim == 2 and y_val.shape[1] > 1:
                    acc = np.mean(np.argmax(val_preds, axis=1) == np.argmax(y_val, axis=1)) * 100
                    val_info = f" | Val Loss: {val_loss:.4f} | Val Acc: {acc:.2f}%"
                else:
                    val_info = f" | Val Loss: {val_loss:.4f}"
                    
            if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
                print(f"Epoch {epoch:3d}/{epochs} - Train Loss: {epoch_loss:.4f}{val_info}")

    def predict(self, X):
        return self.forward(X)