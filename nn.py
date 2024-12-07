import numpy as np 

# For this experiment, all neural networks will be densely connected, without dropout to remove the chance
# of the neurons being initiated favorably

np.set_printoptions(threshold=np.inf)

def init_population(input_size, dense_layers):
    '''
    Creates the information for the population, creates a numpy array that creates a dense layer, which has one list of weights and another 
    list of equal length of biases. One last weight list is also included to multiply nodes by before the output is generated. 

    The input_size is the size of the list that will be inputted into the neural network. The size for this neural network is 
    150, with the 151st element being the answer that the machine learning learning algorithm is trying to predict. 
    '''

    # Creates a very big numpy array with number of dense layers, the width of the array is the input size that is put into the neural netowrk
    # This is a dense neural network, and all layers and denser to reduce the complexity of the nn design and reduce the chance of one off improvements
    # that may occur with favorable/infavorable dropout layers
    chromosome_weights = np.random.randn(dense_layers * 2 + 1, input_size)

    return chromosome_weights

def return_sigmoid(x):
    '''
    Creates the sigmoid shape, which will determine the certainty of the neural network in its output. 

    x is the variable passed into the equation. (sigmoid follows a funny s pattern)
    '''

    # Returns the sigmoid number 
    return 1 / (1+np.exp(-x))

def relu(x):
    ''' 
    Creates a relu shape, in which all numbers are made into 0 and only positive numbers are allowed through 

    x is the variable passed into the equation (follows a path similar to a hockey stick) 
    '''

    # ReLu takes your two numbers and returns the larger one 
    return max(0, x)

def forward_pass_block(input_information, everything_list, current_layer):
    '''
    Evaluates the input information after it has passed through one dense layer (with weights and biases) 

    input_information is the information from before the current dense layer has interacted with it 

    everything_list is the list with all the information for the weights and biases (it was created in the initiation phase) 
    '''

    # Multiplies every weight with its corresponding 
    weights_applied = np.dot(everything_list[current_layer], input_information)
    weights_sums_applied = np.add(everything_list[current_layer + 1], weights_applied)
    print(weights_sums_applied)

    return sum(weights_sums_applied)

def eval(output):
    pass

def fitness(output, actual):
    pass

def evaluate(): 
    pass

init_population(151, 3)