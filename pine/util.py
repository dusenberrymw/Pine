'''
Created on Sept 9, 2014

@author: dusenberrymw
'''
import math


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
        computed_output_vector = network.compute_network_output(example[1])
        for target_output, computed_output in \
                zip(example[0], computed_output_vector):
            residual = target_output - computed_output
            error += residual*residual  # square the residual value
            num_values += 1 # keep count of number of value
    # average the error and take the square root
    return math.sqrt(error/num_values)


def print_network_error(network, training_data, testing_data):
    """Print the current error for the given network"""
    error = calculate_RMS_error(network, training_data)
    print('Error w/ Training Data: {0}'.format(error))
    error = calculate_RMS_error(network, testing_data)
    print('Error w/ Test Data: {0}'.format(error))


def print_network_outputs(network, testing_data):
    """Print the given network's outputs on the test data"""
    for i in range(len(testing_data)):
        outputs = network.compute_network_output(testing_data[i][1])
        print('Input: {0}, Target Output: {1}, Actual Output: {2}'.
              format(testing_data[i][1], testing_data[i][0],
                     outputs))
