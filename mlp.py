import numpy as np
import pandas as pd

# TODO: This does not work

# Activation Functions


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)

# Mean Squared Error Loss


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Performance Metrics


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# Multi-Layer Perceptron


class MLP:
    def __init__(self, input_size, hidden_size, output_size, activation):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialise weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

        # Select activation function
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
        else:
            raise ValueError("Unsupported activation function")

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.activation(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, learning_rate):
        m = X.shape[0]

        # Compute gradients
        dL_da2 = (self.a2 - y) / m  # Derivative of loss
        dL_dz2 = dL_da2 * sigmoid_derivative(self.a2)
        dL_dW2 = np.dot(self.a1.T, dL_dz2)
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)

        dL_da1 = np.dot(dL_dz2, self.W2.T)
        dL_dz1 = dL_da1 * self.activation_derivative(self.a1)
        dL_dW1 = np.dot(X.T, dL_dz1)
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)

        # Update weights and biases
        self.W2 -= learning_rate * dL_dW2
        self.b2 -= learning_rate * dL_db2
        self.W1 -= learning_rate * dL_dW1
        self.b1 -= learning_rate * dL_db1

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = mse(y, y_pred)
            self.backward(X, y, learning_rate)

            # if epoch % 50 == 0:
            #     print(f'Epoch {epoch}, Loss: {loss:.4f}')

    def evaluate(self, X, y):
        y_pred = self.forward(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        print(f'Mean Squared Error: {mse:.4f}, R2 Score: {r2:.4f}')


if __name__ == "__main__":
    df = pd.read_csv('data.csv')
    X = df.iloc[:, :-1].values.astype(np.float64)
    y = df.iloc[:, -1].values.reshape(-1, 1).astype(np.float64)
    input_size = X.shape[1]
    hidden_size = 128
    output_size = 1

    mlp = MLP(input_size, hidden_size, output_size, activation='sigmoid')
    mlp.train(X, y, epochs=1000, learning_rate=0.1)

    mlp.evaluate(X, y)
