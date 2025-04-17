import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import alive_progress


def sigmoid(x):
    # Clip values to avoid overflow
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.power(x, 2)


def mse(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))


def mean_squared_error(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))


def r2_score(y_true, y_pred):
    ss_res = np.sum(np.square(y_true - y_pred))
    ss_tot = np.sum(np.square(y_true - np.mean(y_true)))
    if ss_tot == 0:
        return 0
    return 1 - (ss_res / (ss_tot + 1e-10))


class MLP:
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu'):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        # Initialize lists to store weights, biases, and activations
        self.weights = []
        self.biases = []
        self.activations = []
        self.z_values = []

        # Input layer to first hidden layer
        prev_size = input_size
        for hidden_size in hidden_sizes:
            # He initialization for ReLU
            if activation == 'relu':
                scale = np.sqrt(2. / prev_size)
            else:
                # Xavier initialization for sigmoid/tanh
                scale = np.sqrt(1. / prev_size)

            self.weights.append(np.random.randn(
                prev_size, hidden_size) * scale)
            self.biases.append(np.zeros((1, hidden_size)))
            prev_size = hidden_size

        # Last hidden layer to output layer
        scale = np.sqrt(1. / prev_size)  # Always use Xavier for output layer
        self.weights.append(np.random.randn(prev_size, output_size) * scale)
        self.biases.append(np.zeros((1, output_size)))

        # Select activation function
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_derivative = tanh_derivative
        else:
            raise ValueError("Unsupported activation function")

    def forward(self, X):
        self.activations = [X]
        self.z_values = []

        # Forward propagation through hidden layers
        for i in range(len(self.weights) - 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            # Clip values to prevent explosion
            z = np.clip(z, -500, 500)
            self.z_values.append(z)
            self.activations.append(self.activation(z))

        # Output layer
        z_out = np.dot(self.activations[-1],
                       self.weights[-1]) + self.biases[-1]
        z_out = np.clip(z_out, -500, 500)
        self.z_values.append(z_out)
        self.activations.append(sigmoid(z_out))

        return self.activations[-1]

    def backward(self, X, y, learning_rate, batch_size):
        m = batch_size
        eps = 1e-8  # Small constant to prevent division by zero

        # Initialize gradients
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]

        # Output layer error
        delta = (self.activations[-1] - y) / (m + eps)

        # Backpropagate through layers
        for l in range(len(self.weights) - 1, -1, -1):
            # Compute gradients
            dW[l] = np.dot(self.activations[l].T, delta)
            db[l] = np.sum(delta, axis=0, keepdims=True)

            # Gradient clipping
            dW[l] = np.clip(dW[l], -1, 1)
            db[l] = np.clip(db[l], -1, 1)

            if l > 0:
                delta = np.dot(
                    delta, self.weights[l].T) * self.activation_derivative(self.activations[l])
                # Clip delta to prevent explosion
                delta = np.clip(delta, -1, 1)

        # Update weights and biases with momentum
        if not hasattr(self, 'velocity_W'):
            self.velocity_W = [np.zeros_like(w) for w in self.weights]
            self.velocity_b = [np.zeros_like(b) for b in self.biases]

        momentum = 0.9
        for l in range(len(self.weights)):
            self.velocity_W[l] = momentum * \
                self.velocity_W[l] - learning_rate * dW[l]
            self.velocity_b[l] = momentum * \
                self.velocity_b[l] - learning_rate * db[l]

            self.weights[l] += self.velocity_W[l]
            self.biases[l] += self.velocity_b[l]

    def train(self, X, y, epochs, learning_rate, batch_size=32, validation_split=0.2):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split)

        n_samples = X_train.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))

        train_losses = []
        val_losses = []

        with alive_progress.alive_bar(epochs) as bar:
            for epoch in range(epochs):
                # Shuffle training data
                indices = np.random.permutation(n_samples)
                X_shuffled = X_train[indices]
                y_shuffled = y_train[indices]

                epoch_losses = []

                # Mini-batch training
                for i in range(n_batches):
                    start_idx = i * batch_size
                    end_idx = min(start_idx + batch_size, n_samples)

                    X_batch = X_shuffled[start_idx:end_idx]
                    y_batch = y_shuffled[start_idx:end_idx]

                    y_pred = self.forward(X_batch)
                    batch_loss = mse(y_batch, y_pred)
                    epoch_losses.append(batch_loss)

                    self.backward(X_batch, y_batch, learning_rate,
                                  end_idx - start_idx)

                # Calculate losses
                train_pred = self.forward(X_train)
                train_loss = mse(y_train, train_pred)
                train_losses.append(train_loss)

                val_pred = self.forward(X_val)
                val_loss = mse(y_val, val_pred)
                val_losses.append(val_loss)

                bar()

        # return train_losses, val_losses

    def evaluate(self, X, y):
        y_pred = self.forward(X)
        mse_val = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        print(f'Mean Squared Error: {mse_val:.4f}, R2 Score: {r2:.4f}')
        return mse_val, r2


if __name__ == "__main__":
    # Load and preprocess data
    df = pd.read_csv('data.csv')
    X = df.iloc[:, :-1].values.astype(np.float64)
    y = df.iloc[:, -1].values.reshape(-1, 1).astype(np.float64)

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Normalize target variable as well
    y_scaler = StandardScaler()
    y = y_scaler.fit_transform(y)

    input_size = X.shape[1]
    hidden_sizes = [128, 64]  # Smaller network to start
    output_size = 1

    mlp = MLP(input_size, hidden_sizes, output_size, activation='relu')
    train_losses, val_losses = mlp.train(
        X, y, epochs=100, learning_rate=0.01, batch_size=32)

    mlp.evaluate(X, y)
