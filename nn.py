import numpy as np 
from my_secrets import path_location

# For this experiment, all neural networks will be densely connected, without dropout to remove the chance
# of the neurons being initiated favorably

data = np.load(path_location)
# print(data.shape) # 18813 by 141 
'''
The above code loads in the data, the first element in (18813, 151) is how many tests there are, while the 151 is the size of the matrix
The first 150 numbers are all the information about the past 30 days (open, high low, close and volume). (5 times 30 = 150)

The last data point (the 151st) is the average of the 7 next highs and lows. (so it takes 14 values, the next 7 lows and the next 7 highs and finds average) 

The 151st element is the correct value that the neural networks are trying to predict. Most accurate model is the best. 
'''

np.set_printoptions(threshold=np.inf)

def init_population(input_size, dense_layers):
    '''
    Creates the information for the population, creates a numpy array that creates a dense layer, which has one list of weights and another 
    list of equal length of biases. One last weight list is also included to multiply nodes by before the output is generated. 

    The input_size is the size of the list that will be inputted into the neural network. The size for this neural network is 
    150, with the 151st element being the answer that the machine learning learning algorithm is trying to predict. 
    '''

    # Trial code below before I transitioned to a simpler method
    # total_chromosome_weights = []

    # for x in range(dense_layers):

    #     for y in range(input_size):

    #         local_weights = np.random.randn(input_size)
    #         local_biases = np.random.randn(input_size)

    #         total_chromosome_weights.append(local_weights)
    #         total_chromosome_weights.append(local_biases)
    
    # ending_weights = np.random.rand(input_size)
    # ending_biases = np.random.rand(input_size)

    # # Append the ending weights and biases as separate elements
    # total_chromosome_weights.append(ending_weights)
    # total_chromosome_weights.append(ending_biases)

    # total_chromosome_weights_array = np.array(total_chromosome_weights)

    # print(total_chromosome_weights_array.shape)
    
    total_chromosome_weights_array = np.random.normal(size=(input_size * dense_layers + dense_layers + 2, input_size), loc=0, scale=0.2)
    return total_chromosome_weights_array

def relu(x):
    ''' 
    Creates a relu shape, in which all numbers are made into 0 and only positive numbers are allowed through 

    x is the variable passed into the equation (follows a path similar to a hockey stick) 
    '''

    # ReLu takes your two numbers and returns the larger one 
    return max(0, x)

def forward_pass_block(input_information, everything_list, current_layer, dense_layers):
    '''
    Evaluates the input information after it has passed through one dense layer (with weights and biases) 

    input_information is the information from before the current dense layer has interacted with it 

    everything_list is the list with all the information for the weights and biases (it was created in the initiation phase) 
    '''
    thingy = input_information
    # print(thingy)
    # print(len(thingy))

    # print("HERE", len(input_information))
    numbers = [current_layer * len(input_information), 
               (current_layer + 1) * len(input_information), 
                -dense_layers + current_layer - 2]
    
    input_information = np.array(input_information)

    # Multiplies every weight with its corresponding 
    weights_applied = np.dot(everything_list[numbers[0]:numbers[1]], input_information)

    # REALLY SUSPICIOUS INDEXING
    weights_sums_applied = np.add(everything_list[numbers[2]], weights_applied)
    
    weights_sums_applied = [relu(x) for x in weights_sums_applied]
    weights_sums_applied = np.array(input_information)
    # print(type(weights_sums_applied))
    # print(weights_sums_applied.shape)

    return weights_sums_applied

def final_pass(input_information, everything_list):
    weights_applied = np.dot(everything_list[-2], input_information)
    weights_sums_applied = weights_applied + everything_list[-1][0]

    return weights_sums_applied


def eval(output, actual):
    pass

def fitness(output, actual):
    difference = abs(output - actual)
    return difference ** 2

def evaluate(): 
    pass

input_size = 150
dense_layers = 4
current_number = 0


current_configuration = init_population(input_size=input_size, dense_layers=dense_layers)

new_inputs = data[current_number][0:150]
correct = data[current_number][-1]

for layers in range(0, dense_layers):
    print(layers, type(new_inputs), new_inputs.shape)
    new_inputs = forward_pass_block(input_information=new_inputs, everything_list=current_configuration,current_layer=layers, dense_layers=dense_layers)

final_verdict = final_pass(new_inputs, everything_list=current_configuration)
print("ONE", final_verdict)
print("TWO", correct)
error = fitness(final_verdict, correct)
print(error)