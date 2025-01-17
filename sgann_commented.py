import numpy as np
import matplotlib.pyplot as plt
from my_secrets import path_location  # Import the file path for loading data from a secure module
from sklearn.model_selection import train_test_split  # Import function to split data into training and testing sets
import time  # Import time module to measure execution time
import psutil  # Import psutil to monitor memory usage during execution

def sigmoid(x):
    # Sigmoid activation function: maps input 'x' to a value between 0 and 1
    return 1 / (1 + np.exp(-x))

def relu(x):
    # ReLU activation function: returns 'x' if positive, otherwise returns 0
    return np.maximum(0, x)

def initialize_weights(layers):
    # Initialize weights and biases for each layer in the neural network
    weights = []  # List to store weights for each layer
    biases = []  # List to store biases for each layer
    for i in range(len(layers) - 1):
        # Initialize weights with a normal distribution, scaled by the size of the layers
        weights.append(np.random.normal(size=(layers[i + 1], layers[i]), 
                     scale=np.sqrt(2 / (layers[i] + layers[i + 1]))))
        # Initialize biases with a normal distribution, scaled by 0.5
        biases.append(np.random.normal(size=(layers[i + 1], 1), scale=0.5))
    return weights, biases

def forward_propagation(X, weights, biases):
    # Perform forward propagation through the network
    activations = [X]  # Store activations for each layer, starting with input
    for i in range(len(weights) - 1):
        # Calculate the linear combination of inputs and weights, add bias
        z = np.dot(weights[i], activations[-1]) + biases[i]
        # Apply ReLU activation function
        a = relu(z)
        activations.append(a)
    # Final layer uses sigmoid activation
    z = np.dot(weights[-1], activations[-1]) + biases[-1]
    a = sigmoid(z)
    activations.append(a)
    return activations  # Return all activations, including the output of the final layer

def calculate_loss(X, y, weights, biases):
    # Calculate the loss between predicted and actual values
    predictions = forward_propagation(X, weights, biases)[-1]  # Get the final output
    return np.mean(np.abs(y - predictions))  # Use mean absolute error as loss

def get_genome(weights, biases):
    # Flatten weights and biases into a single genome array for genetic algorithm
    return np.concatenate([w.flatten() for w in weights] + [b.flatten() for b in biases])

def set_genome(genome, layers):
    # Convert a genome array back into weights and biases
    index = 0
    new_weights = []  # List to store new weights
    new_biases = []  # List to store new biases
    
    for i in range(len(layers) - 1):
        # Extract weights for each layer from the genome
        w_size = layers[i + 1] * layers[i]
        w = genome[index:index + w_size].reshape((layers[i + 1], layers[i]))
        new_weights.append(w)
        index += w_size
        
    for i in range(len(layers) - 1):
        # Extract biases for each layer from the genome
        b_size = layers[i + 1]
        b = genome[index:index + b_size].reshape((b_size, 1))
        new_biases.append(b)
        index += b_size
        
    return new_weights, new_biases

def crossover(parent1, parent2, layers):
    # Perform single-point crossover between two parent genomes
    genome1 = get_genome(*parent1)
    genome2 = get_genome(*parent2)
    
    # Randomly select a crossover point
    point = np.random.randint(len(genome1))
    # Create a child genome by combining parts of both parents
    child_genome = np.concatenate([genome1[:point], genome2[point:]])
    
    return set_genome(child_genome, layers)  # Convert child genome back to weights and biases

def mutate(individual, layers, mutation_rate=0.01):
    # Apply mutation to an individual's genome
    genome = get_genome(*individual)
    # Create a mask for mutation based on the mutation rate
    mask = np.random.random(genome.shape) < mutation_rate
    # Apply random normal noise to the selected genes
    genome[mask] += np.random.normal(0, 0.1, np.sum(mask))
    return set_genome(genome, layers)  # Convert mutated genome back to weights and biases

def backpropagation(X, y, weights, biases):
    # Perform backpropagation to compute gradients of weights and biases
    activations = forward_propagation(X, weights, biases)  # Get all activations
    deltas = [activations[-1] - y]  # Compute initial delta for output layer
    
    for i in range(len(weights) - 1, 0, -1):
        # Compute delta for each layer using the derivative of ReLU
        delta = np.dot(weights[i].T, deltas[-1]) * (activations[i] > 0)
        deltas.append(delta)
    
    deltas.reverse()  # Reverse deltas to match the order of layers
    
    grad_weights = []  # List to store gradients of weights
    grad_biases = []  # List to store gradients of biases
    
    for i in range(len(weights)):
        # Compute gradient for weights
        grad_weights.append(np.dot(deltas[i], activations[i].T) / X.shape[1])
        # Compute gradient for biases
        grad_biases.append(np.mean(deltas[i], axis=1, keepdims=True))
    
    return grad_weights, grad_biases

def sgd_update(weights, biases, grad_weights, grad_biases, learning_rate):
    # Update weights and biases using Stochastic Gradient Descent (SGD)
    new_weights = [w - learning_rate * gw for w, gw in zip(weights, grad_weights)]
    new_biases = [b - learning_rate * gb for b, gb in zip(biases, grad_biases)]
    return new_weights, new_biases

def evolve(X, y, population, layers, generations=100, patience=40, batch_size=32, learning_rate=0.15):
    # Evolve the population over a number of generations using a genetic algorithm
    loss_history = []  # Track the best loss over generations
    avg_loss_history = []  # Track the average loss for each generation
    best_loss_history = []  # Track the best loss for each generation after SGD
    best_loss = float('inf')  # Initialize best loss as infinity
    patience_counter = 0  # Counter for early stopping
    no_improvement_counter = 0  # Counter for lack of improvement
    max_memory_usage = 0  # Track maximum memory usage

    process = psutil.Process()  # Get the current process for memory monitoring
    
    for gen in range(generations):
        # Shuffle data and create minibatches
        indices = np.random.permutation(X.shape[1])
        X_shuffled = X[:, indices]
        y_shuffled = y[:, indices]
        
        generation_losses = []  # Track losses for the current generation
        
        for start in range(0, X.shape[1], batch_size):
            end = start + batch_size
            X_batch = X_shuffled[:, start:end]
            y_batch = y_shuffled[:, start:end]
            
            # Evaluate loss for each individual in the population
            losses = sorted([calculate_loss(X_batch, y_batch, *individual) for individual in population])

            avg_loss = np.mean(losses)  # Calculate average loss for the batch
            avg_loss_history.append(avg_loss)

            generation_losses.extend(losses)
            current_best_loss = min(losses)  # Find the best loss in the current batch
            
            if current_best_loss < best_loss - 0.005:
                # Update best loss if improvement is significant
                best_loss = current_best_loss
                patience_counter = 0
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1
                if no_improvement_counter >= 100:
                    patience_counter += 1
            
            if patience_counter >= patience:
                # Early stopping if no improvement for a number of generations
                print(f'Early stopping at generation {gen}, Best Loss: {best_loss}')
                avg_loss_history.append(np.mean(generation_losses))
                return loss_history, avg_loss_history, best_loss_history, max_memory_usage
            
            loss_history.append(best_loss)
            best_loss_history.append(current_best_loss)
            
            # Select parents using roulette wheel selection based on inverse loss
            losses = np.array(losses)
            probabilities = 1 / (losses + 1e-8)  # Add small value to avoid division by zero
            probabilities /= probabilities.sum()
            new_population = []
            
            # Elitism: keep the best individual from the current population
            best_individual = population[np.argmin(losses)]
            new_population.append(best_individual)
            
            while len(new_population) < len(population):
                # Randomly select two parents based on their probabilities
                parents = np.random.choice(len(population), 2, p=probabilities)
                parent1 = population[parents[0]]
                parent2 = population[parents[1]]
                
                # Create a child through crossover and mutation
                child = crossover(parent1, parent2, layers)
                child = mutate(child, layers)
                new_population.append(child)
            
            population = new_population  # Update the population with new individuals
            
            # Apply a very short SGD to all individuals
            for i in range(len(population)):
                grad_weights, grad_biases = backpropagation(X_batch, y_batch, *population[i])
                population[i] = sgd_update(*population[i], grad_weights, grad_biases, learning_rate * 0.1)
            
            # Check current memory usage
            current_memory_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB
            if current_memory_usage > max_memory_usage:
                max_memory_usage = current_memory_usage
            
            print(f'Generation {gen}, Best Loss: {best_loss}, Memory Usage: {current_memory_usage:.2f} MB')

        avg_loss_history.append(np.mean(generation_losses))
        
    return loss_history, avg_loss_history, best_loss_history, max_memory_usage

# Load and prepare data from the specified file path
data = np.load(path_location)  # Load data from the file path specified in 'my_secrets'
X = data[:, 0:150].T  # Extract features and transpose for correct shape
y = data[:, 150].reshape(1, -1)  # Extract labels and reshape

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X.T, y.T, test_size=0.2)
X_train, X_test, y_train, y_test = X_train.T, X_test.T, y_train.T, y_test.T

# Initialize and run genetic algorithm
start_time = time.time()
layers = [150, 150, 150, 150, 150, 1]
population_size = 600
population = [initialize_weights(layers) for _ in range(population_size)]
loss_history, avg_loss_history, best_loss_history, max_memory_usage = evolve(X_train, y_train, population, layers, generations=1000, patience=10, batch_size=30, learning_rate=0.01)

print("Max memory used", max_memory_usage)

end_time = time.time()
print(f'Training time: {end_time - start_time} seconds')

# Plot loss history
plt.plot(loss_history, label='Best Loss')
min_loss = min(loss_history)
plt.axhline(y=min_loss, color='r', linestyle='--', label=f'Lowest Loss: {min_loss:.4f}')
plt.plot(avg_loss_history, color='g', alpha=0.2, label='Average Loss for Generation')  # Faint green line
plt.plot(best_loss_history, color='b', label='Best Loss for each Generation after SGD')  # Best loss for SGD
plt.legend()
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()
