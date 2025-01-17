import numpy as np 
from sklearn.model_selection import train_test_split
from my_secrets import path_location
import matplotlib.pyplot as plt

# For this experiment, all neural networks will be densely connected, without dropout to remove the chance
# of the neurons being initiated favorably (keeping initation the same for all the optimization algorithms, and seeing where they go from there)

np.set_printoptions(threshold=np.inf)

data = np.load(path_location)

# print(data.shape) # 18813 by 141 
'''
The above code loads in the data, the first element in (18813, 151) is how many tests there are, while the 151 is the size of the matrix
The first 150 numbers are all the information about the past 30 days (open, high low, close and volume). (5 times 30 = 150)

The last data point (the 151st) is the average of the 7 next highs and lows. (so it takes 14 values, the next 7 lows and the next 7 highs and finds average) 

The 151st element is the correct value that the neural networks are trying to predict. Most accurate model is the best. 
'''


def init_population(input_size, dense_layers):
    '''
    Creates the information for the population, creates a numpy array that creates a dense layer, which has one list of weights and another 
    list of equal length of biases. One last weight list is also included to multiply nodes by before the output is generated. 

    The input_size is the size of the list that will be inputted into the neural network. The size for this neural network is 
    150, with the 151st element being the answer that the machine learning learning algorithm is trying to predict. 
    '''

    total_chromosome_weights_array = np.random.normal(size=(input_size * dense_layers + dense_layers + 1, input_size), loc=100, scale=20)
    return total_chromosome_weights_array

def forward_pass_block(input_information, everything_list, dense_layers):
    '''
    Evaluates the input information after it has passed through one dense layer (with weights and biases) 

    input_information is the information from before the current dense layer has interacted with it 

    everything_list is the list with all the information for the weights and biases (it was created in the initiation phase) 
    '''

    # this list will be transformed into a numpy list later, it contains all the unactvated information, and the activated 
    complete_for_sample = []

    # This just tells us how many nodes are in each layer, important so it is multiplied by the correct number 
    size_of_input = len(input_information)

    # Stores information in here so the actual input_information is altered (I think this isn't important, not too sure though)
    thingy = input_information

    # Iterates through all the layers 
    for layer in range(dense_layers):
        numbers = [layer * size_of_input, 
                (layer + 1) * size_of_input, 
                    -dense_layers + layer - 1]

        # Uses dot product to evaluate the sum of all the nodes 
        weights_sums_applied = np.dot(thingy, everything_list[numbers[0]:numbers[1]])

        # Applies bias onto all the nodes before it is activated
        weights_sums_applied = weights_sums_applied + everything_list[numbers[2]]

        # Appends the unactivated nodes to a list, this is important layer for the net_k variables
        complete_for_sample.append(weights_sums_applied)

        # Activates everything 
        weights_sums_applied = [sigmoid(x) for x in weights_sums_applied] 

        # This is the information that is going into the next layer, I could use complete_for_sample[-1] but I am a bit scared to optimize cas everything might break
        input_information = weights_sums_applied

        # Appends all the activated data to the next layer. 
        complete_for_sample.append(input_information)

    complete_for_sample = np.array(complete_for_sample)

    return complete_for_sample

def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s

def derivative_sigmoid(x):
    s = sigmoid(x)
    ds = s * (1-s)
    return ds

def final_pass(input_information, everything_list):
    weights_applied = np.dot(everything_list[-2], input_information)
    weights_sums_applied = weights_applied + everything_list[-1][0]

    return weights_sums_applied

def eval(output, actual):
    pass

def fitness(output, actual):
    difference = abs(output - actual)
    return (0.5 * (difference ** 2))

def fitness_derivative_wrt_actual(output, actual):
    to_return = actual - output
    return to_return

def backpropagation_final_layer(current_configuration, learning_rate, output, actual):
    node_gradient = fitness_derivative_wrt_actual(output=output, actual=actual) 

def backpropagation(absolute_error, hidden_activations_and_summations, current_configuration, dense_layers, size_of_input, learning_rate):
    pass 
    gradient_array_for_sample = np.empty_like(current_configuration)
    previous_nodes = np.empty(size_of_input)

    partial_L = absolute_error

    gradient_array_for_sample[-5] = [partial_L * hidden_activations_and_summations[-1][x] for x in range(size_of_input)]
    
    previous_nodes = [partial_L * hidden_activations_and_summations[-1][x] * current_configuration[-5][x] for x in range(size_of_input)]



    # for layer in range(-1, -dense_layers-1, -1):
    #     numbers = [layer * size_of_input, 
    #             (layer + 1) * size_of_input, 
    #                 -dense_layers + layer - 1]

    #     if layer == dense_layers:
    #         error = target - complete_for_sample[-1]
    #         deltas = [error * derivative_sigmoid(complete_for_sample[-1])]
    #     else:
    #         delta = deltas[-1].dot(everything_list[layer * size_of_input:(layer + 1) * size_of_input].T)
    #         delta = delta * derivative_sigmoid(complete_for_sample[layer * 2 - 1])
    #         deltas.append(delta)
    


def update_weights(w, gradient, learning_rate):
    return w - learning_rate * gradient

X = data[:, 0:150]
y = data[:, 150]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

total_squared_error = 0

input_size = 150
learning_rate = 0.01
dense_layers = 4
batch_size = 1
epochs = 100
current_configuration = init_population(input_size=input_size, dense_layers=dense_layers)
# print("You shit", current_configuration.shape)

progressive_error = []

for epoch in range(1, epochs+1):
    total_mse_for_epoch = 0

    selected_batch = np.random.choice(len(X_train), size=batch_size, replace=False)

    X_train_sample = X_train[selected_batch]
    y_train_sample = y_train[selected_batch]
    
    for sample in range(batch_size):
        hidden_layers = forward_pass_block(X_train_sample[sample], current_configuration, dense_layers=dense_layers)
        response = final_pass(hidden_layers[-1], current_configuration)

        sample_fitness = fitness(output=response, actual=y_train_sample[sample])
        total_mse_for_epoch += sample_fitness
        print(current_configuration.shape)

        absolute_error = abs(y_train_sample[sample] -  response)
        # BACKPROPAGATION TIME
        weight_changes_for_sample = backpropagation(absolute_error=absolute_error, hidden_activations_and_summations=hidden_layers, current_configuration=current_configuration, dense_layers=dense_layers, size_of_input=input_size, learning_rate=learning_rate)

    # updated_weights = update_weights(current_configuration, weight_changes_for_sample)

    # current_configuration = updated_weights




    average = total_mse_for_epoch / batch_size
    progressive_error.append(average)

    print(f"Epoch {epoch}: MSE {average}")

plt.plot(progressive_error)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show() 