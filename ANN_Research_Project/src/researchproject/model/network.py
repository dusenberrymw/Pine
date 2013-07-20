'''
Created on Jun 6, 2013

@author: dusenberrymw
'''
import random
import math


class Network(object):
    """A class for the overall network"""
    
    def __init__(self, num_inputs, num_neurons_list, activation_functions):
        """Constructor
        
        'num_neurons_list' = list containing the number of neurons in each
                                layer, starting with the first hidden layer, 
                                since we do not make objects for the input
                                layer.
        'activation_functions' = list containing the activation functions for
                                    each layer
        """
        self.layers = []
        for num_neurons, activation_function in \
                zip(num_neurons_list, activation_functions):
            if num_neurons is None:
                # Then automatically determine number of neurons
                # -general rules of thumb:
                #     -The number of hidden neurons should be between the size
                #        of the input layer and the size of the output layer.
                #     -The number of hidden neurons should be 2/3 the size of 
                #        the input layer, plus the size of the output layer.
                #     -The number of hidden neurons should be less than twice 
                #        the size of the input layer.
                num_neurons = (round((2/3) * num_inputs) + 
                               num_neurons_list[-1])
            self.layers.append(Layer(num_neurons, num_inputs, 
                                     activation_function))
            num_inputs = num_neurons # for the next layer
    
    def compute_network_output(self, inputs):
        """Compute output(s) of network given one entry (row) of data"""
        for layer in self.layers:
            activate = layer.activation_function.activate
            outputs = []
            # the first layer is the hidden neurons,
            #    so the inputs are those supplied to the
            #    network
            # then, for the next layers, the inputs will
            #    be the outputs of the previous layer
            for neuron in layer.neurons:
                # keep track of what inputs were sent to this neuron
                neuron.inputs = inputs
                # multiply each input with the associated weight for that connection
                local_output = 0.0  
                for input_value, weight_value in zip(inputs, neuron.weights):
                    local_output += input_value * weight_value 
                # then subtract the threshold value
                local_output -= neuron.threshold
                # finally, use the activation function to determine the output
                local_output = activate(local_output)
                # store outputs
                neuron.local_output = local_output
                outputs.append(local_output)  
            # the inputs to the next layer will be the outputs
            #    of the previous layer
            inputs = outputs   
        return outputs
    
    def calculate_error(self, inputs, target_outputs):
        """Determine the root mean square (RMS) error for the given dataset
        (multiple rows) of input data against the associated target outputs
        
        RMS error is the square root of: the sum of the squared differences 
        between target outputs and actual outputs, divided by the total number
        of values.  It is useful because it eliminates the need to take 
        absolute values, which would be necessary otherwise to prevent two 
        opposite errors from canceling out.
        
        error = sqrt( (sum(residual^2)) / num_values )
        
        """
        error = 0.0
        num_values = 0
        # for each row of data
        for input_set, target_output_set in zip(inputs, target_outputs):
            computed_output_set = self.compute_network_output(input_set)
            for target_output, computed_output in \
                    zip(target_output_set, computed_output_set):
                residual = target_output - computed_output
                error += residual*residual  # square the residual value
                num_values += 1 # keep count of number of value
        # average the error and take the square root
        return math.sqrt(error/num_values)


class Layer(object):
    """A class for layers in the network"""
    
    def __init__(self, num_neurons, num_inputs, activation_function):
        """Constructor"""
        self.activation_function = activation_function
        self.neurons = [Neuron(num_inputs) for _ in range(num_neurons)]


class Neuron(object):
    """A class for neurons in the network"""
    
    def __init__(self, num_inputs):
        """Constructor"""
        self.inputs = [] # the inputs coming from previous neurons
        self.local_output = 0.0 # the output leaving this neuron
        # need a weight for each input to the neuron
        self.weights = [random.uniform(-0.9,0.9) for _ in range(num_inputs)]
        self.threshold = random.uniform(-0.9,0.9)
        
        # REMOVE THESE at some point
        self.prev_weight_deltas = [0.1]*num_inputs
        self.prev_threshold_delta = 0.1
        self.partial_weight_gradients = [0]*num_inputs
        self.partial_threshold_gradient = 0
        self.prev_partial_weight_gradients = [0]*num_inputs
        self.prev_partial_threshold_gradient = 0
    
    def compute_output(self, inputs, activation_function):
        """Given a set of inputs from previous layer neuron,
        will compute the local output of the neuron
        """
        # keep track of what inputs were sent to this neuron
        self.inputs = inputs
        # multiply each input with the associated weight for that connection
        local_output = 0.0 
        for input_value, weight_value in zip(inputs, self.weights):
            local_output += input_value * weight_value
        # then subtract the threshold value
        local_output -= self.threshold
        # finally, use the activation function to determine the output
        local_output = (activation_function.
            activate(local_output))
        # store outputs
        self.local_output = local_output
        return local_output
    

def print_network_error(network, data):
    """Print the current error for the given network"""
    error = network.calculate_error(data.training_inputs, 
                                    data.training_target_outputs)   
    print('Error w/ Training Data: {0}'.format(error))
    error = network.calculate_error(data.testing_inputs, 
                                    data.testing_target_outputs)   
    print('Error w/ Test Data: {0}'.format(error))

    
def print_network_outputs(network, data):
    """Print the given network's outputs on the test data"""
    for i in range(len(data.testing_inputs)):
        outputs = network.compute_network_output(data.testing_inputs[i])
        print('Input: {0}, Target Output: {1}, Actual Output: {2}'.
              format(data.testing_inputs[i], data.testing_target_outputs[i], 
                     outputs))

