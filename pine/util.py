'''
Created on Sept 9, 2014

@author: dusenberrymw
'''
import math
import pine.network


def calculate_RMS_error(network, examples):
    """Determine the root mean square (RMS) error of the network for the given
    dataset (multiple rows) of input data against the associated target outputs

    RMS error is the square root of: the sum of the squared differences
    between target outputs and actual outputs, divided by the total number
    of values.  It is useful because it eliminates the need to take
    absolute values, which would be necessary otherwise to prevent two
    opposite errors from canceling out.

    error = sqrt( (sum(residual^2)) / num_values )

    examples are in the format: [[target_vector], [input_vector]]

    """
    # error = 0.0
    # num_values = 0
    # # for each row of data
    # for example in examples:
    #     computed_output_vector = network.forward(example[1])
    #     for target_output, computed_output in \
    #             zip(example[0], computed_output_vector):
    #         residual = target_output - computed_output
    #         error += residual*residual  # square the residual value
    #         num_values += 1 # keep count of number of value
    # # average the error and take the square root
    # return math.sqrt(error/num_values)
    return calculate_average_cost_legacy(network, examples)

def calculate_average_cost_legacy(network, examples):
    target_vectors = [example[0] for example in examples]
    input_vectors = [example[1] for example in examples]
    return calculate_average_cost(network, target_vectors, input_vectors)



def calculate_average_cost(network, target_vectors, input_vectors):
    """
    Calculate the network's average cost over the given examples

    """
    cost_vector = []
    for target_vector, input_vector in zip(target_vectors, input_vectors):
        network.forward(input_vector)
        cost = network.cost(target_vector)
        cost_vector.append(cost)
    avg_cost = sum(cost_vector)/len(cost_vector)
    return avg_cost


def create_network(layout, activation_function_names):
    """ Create a new network

    'layout' = list of ints containing the number of neurons in each
                        layer, including input layer
               -General rules of thumb for number of hidden neurons:
                 -The number of hidden neurons should be between the size
                    of the input layer and the size of the output layer.
                 -The number of hidden neurons should be 2/3 the size of
                    the input layer, plus the size of the output layer.
                 -The number of hidden neurons should be less than twice
                    the size of the input layer.

    'activation_function_names' = list of strings containing the name of the
                                      activation function for each hidden and
                                      output layer (skip input layer)

    """
    network = pine.network.Network()
    num_inputs = layout.pop(0) # we don't make objects for input neurons
    for num_neurons, act_func_str in zip(layout, activation_function_names):
        layer = create_layer(num_neurons, num_inputs, act_func_str)
        network.layers.append(layer)
        num_inputs = num_neurons
    return network


def create_layer(num_neurons, num_inputs, activation_function_name):
    """ Create a new layer

    'num_neurons' = number of neurons in this layer

    'num_inputs' = number of inputs being fed into this layer

    'activation_function_name'  = name of the activation function for this layer


    """
    layer = pine.network.Layer()
    act_func_str = activation_function_name.lower()
    layer.neurons = [create_neuron(num_inputs, act_func_str) for _ in range(num_neurons)]
    return layer


def create_neuron(num_inputs, activation_function_name):
    """ Create a new neuron

    'num_inputs' = number of inputs being fed into this neuron

    'activation_function_name' = name of the activation function for this neuron

    """
    if activation_function_name == "logistic":
        act_func = pine.activation.Logistic()
    elif activation_function_name == "tanh":
        act_func = pine.activation.Tanh()
    else:
        act_func = pine.activation.Linear()
    neuron = pine.network.Neuron(num_inputs, act_func)
    return neuron


def isValidFunction(name):
    functionNames = ["logistic", "tanh", "linear"]
    return name.lower() in functionNames


def print_network_error(network, training_data, testing_data):
    """Print the current error for the given network"""
    error = calculate_average_cost_legacy(network, training_data)
    print('Avg cost w/ Training Data: {0}'.format(error))
    error = calculate_average_cost_legacy(network, testing_data)
    print('Avg cost w/ Test Data: {0}'.format(error))


def print_network_outputs(network, testing_data):
    """Print the given network's outputs on the test data"""
    for i in range(len(testing_data)):
        outputs = network.forward(testing_data[i][1])
        print('Input: {0}, Target Output: {1}, Actual Output: {2}'.
              format(testing_data[i][1], testing_data[i][0],
                     outputs))
