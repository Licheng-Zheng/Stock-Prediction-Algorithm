import numpy as np
import matplotlib.pyplot as plt
from my_secrets import path_location
from sklearn.model_selection import train_test_split
import time
import psutil

# Activation functions and their derivatives, used in forward propagation and backpropagation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Initialize weights and biases
def initialize_weights(layers):
    weights = []
    biases = []
    for i in range(len(layers) - 1):
        weights.append(np.random.normal(size=(layers[i + 1], layers[i]), loc=0, scale=np.sqrt(2 / (layers[i] + layers[i + 1]))))
        biases.append(np.random.normal(size=(layers[i + 1], 1), loc=0, scale=0.5))
    return weights, biases

# Forward propagation
def forward_propagation(X, weights, biases):
    activations = [X]
    for i in range(len(weights) - 1):
        z = np.dot(weights[i], activations[-1]) + biases[i]
        a = relu(z)
        activations.append(a)
    z = np.dot(weights[-1], activations[-1]) + biases[-1]
    a = sigmoid(z)
    activations.append(a)
    return activations

# Backward propagation
def backward_propagation(y, activations, weights):
    deltas = [activations[-1] - y]
    for i in reversed(range(len(weights) - 1)):
        delta = np.dot(weights[i + 1].T, deltas[-1]) * relu_derivative(activations[i + 1])
        deltas.append(delta)
    deltas.reverse()
    return deltas

# Update weights and biases
def update_parameters(weights, biases, activations, deltas, learning_rate):
    for i in range(len(weights)):
        weights[i] -= learning_rate * np.dot(deltas[i], activations[i].T)
        biases[i] -= learning_rate * np.mean(deltas[i], axis=1, keepdims=True)

# Load data
data = np.load(path_location)
X = data[:, 0:150].T
y = data[:, 150].reshape(1, -1)

start_time = time.time()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X.T, y.T, test_size=0.2)
X_train, X_test, y_train, y_test = X_train.T, X_test.T, y_train.T, y_test.T

# Initialize parameters
layers = [150, 150, 150, 150, 150, 1]
weights, biases = initialize_weights(layers)
learning_rate = 0.005
epochs = 1000
batch_size = 60
patience = 10
best_error = float('inf')
patience_counter = 0

# Training loop
errors = []
min_error = float('inf')
min_error_epoch = 0

max_memory_usage_mb = 0

for epoch in range(epochs):
    # Shuffle training data
    indices = np.random.permutation(X_train.shape[1])
    X_train, y_train = X_train[:, indices], y_train[:, indices]
    for i in range(0, X_train.shape[1], batch_size):
        X_batch = X_train[:, i:i + batch_size]
        y_batch = y_train[:, i:i + batch_size]
        activations = forward_propagation(X_batch, weights, biases)
        deltas = backward_propagation(y_batch, activations, weights)
        update_parameters(weights, biases, activations, deltas, learning_rate)

    # Validate the model
    activations = forward_propagation(X_test, weights, biases)
    error = np.mean(np.abs(y_test - activations[-1]))
    errors.append(error)
    print(f'Epoch {epoch}, Validation Error: {error}')
    
    # Memory usage by this process
    process = psutil.Process()
    memory_usage_mb = process.memory_info().rss / (1024 ** 2)
    print(f'Memory Usage: {memory_usage_mb:.2f} MB')
    
    if memory_usage_mb > max_memory_usage_mb:
        max_memory_usage_mb = memory_usage_mb

    # Early stopping mechanism
    if error < best_error:
        best_error = error
        min_error = error
        min_error_epoch = epoch
        patience_counter = 0
    else:
        patience_counter += 1
    if patience_counter >= patience:
        print(f'Early stopping at epoch {epoch}')
        break

print(f'Maximum Memory Usage: {max_memory_usage_mb:.2f} MB')

end_time = time.time()

print(f'Training time: {end_time - start_time} seconds')

# Plot errors
plt.plot(errors, label='Validation Error')
plt.axhline(y=min_error, color='r', linestyle='--', label=f'Lowest loss: {min_error:.4f}')
plt.xlabel('Epochs')
plt.ylabel('Validation Error')
plt.legend()
plt.show()
