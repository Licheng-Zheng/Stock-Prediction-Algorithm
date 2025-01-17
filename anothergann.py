import numpy as np
import matplotlib.pyplot as plt
from my_secrets import path_location
from sklearn.model_selection import train_test_split
import time
import psutil

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def initialize_weights(layers):
    weights = []
    biases = []
    for i in range(len(layers) - 1):
        weights.append(np.random.normal(size=(layers[i + 1], layers[i]), 
                     scale=np.sqrt(2 / (layers[i] + layers[i + 1]))))
        biases.append(np.random.normal(size=(layers[i + 1], 1), scale=0.5))
    return weights, biases

def forward_propagation(X, weights, biases):
    activations = [X]
    for i in range(len(weights) - 1):
        z = np.dot(weights[i], activations[-1]) + biases[i]
        a = relu(z)
        activations.append(a)
    z = np.dot(weights[-1], activations[-1]) + biases[-1]
    a = sigmoid(z)
    activations.append(a)
    return activations[-1]

def calculate_loss(X, y, weights, biases):
    predictions = forward_propagation(X, weights, biases)
    return np.mean(np.abs(y - predictions))  # Loss value

def get_genome(weights, biases):
    return np.concatenate([w.flatten() for w in weights] + [b.flatten() for b in biases])

def set_genome(genome, layers):
    index = 0
    new_weights = []
    new_biases = []
    
    for i in range(len(layers) - 1):
        w_size = layers[i + 1] * layers[i]
        w = genome[index:index + w_size].reshape((layers[i + 1], layers[i]))
        new_weights.append(w)
        index += w_size
        
    for i in range(len(layers) - 1):
        b_size = layers[i + 1]
        b = genome[index:index + b_size].reshape((b_size, 1))
        new_biases.append(b)
        index += b_size
        
    return new_weights, new_biases

def crossover(parent1, parent2, layers):
    genome1 = get_genome(*parent1)
    genome2 = get_genome(*parent2)
    
    # Single point crossover
    point = np.random.randint(len(genome1))
    child_genome = np.concatenate([genome1[:point], genome2[point:]])
    
    return set_genome(child_genome, layers)

def mutate(individual, layers, mutation_rate=0.01):
    genome = get_genome(*individual)
    mask = np.random.random(genome.shape) < mutation_rate
    genome[mask] += np.random.normal(0, 0.1, np.sum(mask))
    return set_genome(genome, layers)

def evolve(X, y, population, layers, generations=100, patience=20, batch_size=32):
    loss_history = []
    avg_loss_history = []
    best_loss_in_generation = []
    best_loss = float('inf')
    patience_counter = 0
    no_improvement_counter = 0
    max_memory_usage = 0
    
    
    process = psutil.Process()
    
    for gen in range(generations):
        # Create minibatches
        indices = np.random.permutation(X.shape[1])
        X_shuffled = X[:, indices]
        y_shuffled = y[:, indices]
        
        generation_losses = []
        
        for start in range(0, X.shape[1], batch_size):
            end = start + batch_size
            X_batch = X_shuffled[:, start:end]
            y_batch = y_shuffled[:, start:end]
            
            # Evaluate loss sequentially
            losses = [calculate_loss(X_batch, y_batch, *individual) for individual in population]



            avg_loss = np.mean(losses)
            avg_loss_history.append(avg_loss)

            generation_losses.extend(losses)
            current_best_loss = min(losses)
            
            if current_best_loss < best_loss - 0.005:
                best_loss = current_best_loss
                patience_counter = 0
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1
                if no_improvement_counter >= 100:
                    patience_counter += 1
            
            if patience_counter >= patience:
                print(f'Early stopping at generation {gen}, Best Loss: {best_loss}')
                avg_loss_history.append(np.mean(generation_losses))
                return loss_history, avg_loss_history, max_memory_usage
            
            loss_history.append(best_loss)
            
            # Select parents (roulette wheel selection based on inverse loss)
            losses = np.array(losses)
            probabilities = 1 / (losses + 1e-8)  # Add small value to avoid division by zero
            probabilities /= probabilities.sum()
            new_population = []
            
            # Elitism: keep the best individual
            best_individual = population[np.argmin(losses)]
            new_population.append(best_individual)
            
            while len(new_population) < len(population):
                parents = np.random.choice(len(population), 2, p=probabilities)
                parent1 = population[parents[0]]
                parent2 = population[parents[1]]
                
                # Create child
                child = crossover(parent1, parent2, layers)
                child = mutate(child, layers)
                new_population.append(child)
            
            population = new_population
            
            # Check memory usage
            current_memory_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB
            if current_memory_usage > max_memory_usage:
                max_memory_usage = current_memory_usage
            
            print(f'Generation {gen}, Best Loss: {best_loss}, Memory Usage: {current_memory_usage:.2f} MB')
        
        avg_loss_history.append(np.mean(generation_losses))
        
    return loss_history, avg_loss_history, max_memory_usage

# Load and prepare data
data = np.load(path_location)
X = data[:, 0:150].T
y = data[:, 150].reshape(1, -1)

X_train, X_test, y_train, y_test = train_test_split(X.T, y.T, test_size=0.2)
X_train, X_test, y_train, y_test = X_train.T, X_test.T, y_train.T, y_test.T

# Initialize and run genetic algorithm
start_time = time.time()
layers = [150, 150, 150, 150, 150, 1]
population_size = 600
population = [initialize_weights(layers) for _ in range(population_size)]
loss_history = evolve(X_train, y_train, population, layers, generations=1000, patience=10, batch_size=32)

print("Max memory used", loss_history[2])

end_time = time.time()
print(f'Training time: {end_time - start_time} seconds')

# Get initial memory usage
process = psutil.Process()
initial_memory = process.memory_info().rss / (1024 * 1024)  # Convert to MB
print(f'Initial memory usage: {initial_memory:.2f} MB')

# Plot loss history
plt.plot(loss_history[0], label='Best Loss')
min_loss = min(loss_history[0])
plt.axhline(y=min_loss, color='r', linestyle='--', label=f'Lowest Loss: {min_loss:.4f}')
plt.plot(loss_history[1], color='g', alpha=0.2, label='Average Loss for Generation')  # Faint green line
plt.legend()
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()
