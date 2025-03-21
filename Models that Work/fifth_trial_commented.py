import numpy as np
import matplotlib.pyplot as plt
from my_secrets import path_location
from sklearn.model_selection import train_test_split
import time
import psutil

# Define activation functions and their derivatives for neural network operations
def sigmoid(x):
    # Sigmoid activation function
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    # Derivative of the sigmoid function
    return x * (1 - x)

def relu(x):
    # ReLU activation function
    return np.maximum(0, x)

def relu_derivative(x):
    # Derivative of the ReLU function
    return np.where(x > 0, 1, 0)

# Function to initialize weights and biases for the neural network
def initialize_weights(layers):
    weights = []
    biases = []
    for i in range(len(layers) - 1):
        # Initialize weights with a normal distribution
        weights.append(np.random.normal(size=(layers[i + 1], layers[i]), loc=0, scale=np.sqrt(2 / (layers[i] + layers[i + 1]))))
        # Initialize biases with a normal distribution
        biases.append(np.random.normal(size=(layers[i + 1], 1), loc=0, scale=0.5))

    # The weights and biases of the neural network are returned
    return weights, biases

# Perform forward propagation through the network
def forward_propagation(X, weights, biases):
    activations = [X]
    for i in range(len(weights) - 1):
        # Calculate the linear combination of inputs and weights
        z = np.dot(weights[i], activations[-1]) + biases[i]
        # Apply ReLU activation function
        a = relu(z)
        activations.append(a)
    # Final layer uses sigmoid activation function
    z = np.dot(weights[-1], activations[-1]) + biases[-1]
    a = sigmoid(z)
    activations.append(a)
    return activations

# Perform backward propagation to calculate deltas for weight updates
def backward_propagation(y, activations, weights):
    # Calculate the initial delta for the output layer, the deltas are the gradient, and it is how we update the weights and biases to reduce loss
    deltas = [activations[-1] - y]
    for i in reversed(range(len(weights) - 1)):
        # Calculate delta for each layer using the derivative of the activation function
        delta = np.dot(weights[i + 1].T, deltas[-1]) * relu_derivative(activations[i + 1])
        deltas.append(delta)

    # This has to be reversed because we do backpropagation from back to front, and we apply them front to back, so we have to flip it around
    deltas.reverse()
    return deltas

# Update weights and biases using calculated deltas and learning rate
def update_parameters(weights, biases, activations, deltas, learning_rate):
    for i in range(len(weights)):
        # Update weights
        weights[i] -= learning_rate * np.dot(deltas[i], activations[i].T)
        # Update biases
        biases[i] -= learning_rate * np.mean(deltas[i], axis=1, keepdims=True)

# Load data from the specified file path
data = np.load(path_location)

# Extract features and labels from the data, the elements from 0 to 150 are the input information (x) and the last element (element 150) is the expected output (y)
X = data[:, 0:150].T
y = data[:, 150].reshape(1, -1)

start_time = time.time()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X.T, y.T, test_size=0.2)
X_train, X_test, y_train, y_test = X_train.T, X_test.T, y_train.T, y_test.T

# Define the architecture of the neural network
layers = [150, 150, 150, 150, 150, 1]
# Initialize weights and biases
weights, biases = initialize_weights(layers)

# How quickly does the neural network train 
learning_rate = 0.005

# The maxmimum epochs the neural network can train for (it never reaches this value before patience ends the process) 
epochs = 1000

# Size of batch before optimization for the epoch
batch_size = 100

# How many epochs without improvement before the process ends (this is the stopping criteria)
patience = 40

best_error = float('inf')
patience_counter = 0

# Initialize variables for tracking errors and memory usage, this information is going to be saved and graphed later on 
errors = []
min_error = float('inf')
min_error_epoch = 0
max_memory_usage_mb = 0

# Training loop for the neural network
for epoch in range(epochs):
    # Shuffle training data for each epoch
    indices = np.random.permutation(X_train.shape[1])
    X_train, y_train = X_train[:, indices], y_train[:, indices]
    for i in range(0, X_train.shape[1], batch_size):
        # Create mini-batches for training
        X_batch = X_train[:, i:i + batch_size]
        y_batch = y_train[:, i:i + batch_size]
        # Perform forward and backward propagation
        activations = forward_propagation(X_batch, weights, biases)
        deltas = backward_propagation(y_batch, activations, weights)
        # Update weights and biases
        update_parameters(weights, biases, activations, deltas, learning_rate)

    # Validate the model on the test set
    activations = forward_propagation(X_test, weights, biases)
    error = np.mean(np.abs(y_test - activations[-1]))
    errors.append(error)
    print(f'Epoch {epoch}, Validation Error: {error}')
    
    # Monitor memory usage of the process
    process = psutil.Process()
    memory_usage_mb = process.memory_info().rss / (1024 ** 2)
    print(f'Memory Usage: {memory_usage_mb:.2f} MB')
    
    # Track maximum memory usage
    if memory_usage_mb > max_memory_usage_mb:
        max_memory_usage_mb = memory_usage_mb

    # Implement early stopping based on validation error
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

# Print total training time
print(f'Training time: {end_time - start_time} seconds')

# Plot validation errors over epochs, graph made using matplotlib!
plt.plot(errors, label='Validation Error')
plt.axhline(y=min_error, color='r', linestyle='--', label=f'Lowest loss: {min_error:.4f}')
plt.xlabel('Epochs')
plt.ylabel('Validation Error')
plt.legend()
plt.show()