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
    error = 0.0
    num_values = 0
    # for each row of data
    for example in examples:
        computed_output_vector = network.forward(example[1])
        for target_output, computed_output in \
                zip(example[0], computed_output_vector):
            residual = target_output - computed_output
            error += residual*residual  # square the residual value
            num_values += 1 # keep count of number of value
    # average the error and take the square root
    return math.sqrt(error/num_values)


def cost_J(network, example):
    """Determine the overall cost J(theta) for the network

    The overal cost, J(theta), is the overall "error" of the network
        with respect to parameter (weight) theta, and is equal to the
        sum of the cost functions for each node in the output layer,
        evaluated at the given training example

    examples are in the format: [[target_vector], [input_vector]]

    """
    target_output_vector = example[0]
    input_vector = example[1]
    output_layer = network.layers[-1]
    cost_func = output_layer.activation_function.cost
    hypothesis_vector = network.forward(input_vector)
    return sum([cost_func(hypothesis_vector[i], target_output_vector[i])
                for i in range(len(output_layer.neurons))])


def create_network(layout, activation_functions):
    """

    'layout' = list of ints containing the number of neurons in each layer,
                including input layer
               -General rules of thumb for number of hidden neurons:
                 -The number of hidden neurons should be between the size
                    of the input layer and the size of the output layer.
                 -The number of hidden neurons should be 2/3 the size of
                    the input layer, plus the size of the output layer.
                 -The number of hidden neurons should be less than twice
                    the size of the input layer.

    'activation_functions' = list of strings containing the name of the
                                activation function for each hidden and
                                output layer (skip input layer)

    """
    network = pine.network.Network()
    num_inputs = layout.pop(0) # we don't make objects for input neurons
    for num_neurons, act_func_str in zip(layout, activation_functions):
        if act_func_str.lower() == "logistic":
            act_func = pine.training.LogisticActivationFunction()
        elif act_func_str.lower() == "tanh":
            act_func = pine.training.TanhActivationFunction()
        else:
            act_func = pine.training.LinearActivationFunction()
        network.layers.append(pine.network.Layer(num_neurons, num_inputs, act_func))
        num_inputs = num_neurons
    return network


def print_network_error(network, training_data, testing_data):
    """Print the current error for the given network"""
    error = calculate_RMS_error(network, training_data)
    print('Error w/ Training Data: {0}'.format(error))
    error = calculate_RMS_error(network, testing_data)
    print('Error w/ Test Data: {0}'.format(error))


def print_network_outputs(network, testing_data):
    """Print the given network's outputs on the test data"""
    for i in range(len(testing_data)):
        outputs = network.forward(testing_data[i][1])
        print('Input: {0}, Target Output: {1}, Actual Output: {2}'.
              format(testing_data[i][1], testing_data[i][0],
                     outputs))
