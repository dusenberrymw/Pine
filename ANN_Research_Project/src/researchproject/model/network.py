'''
Created on Jun 6, 2013

@author: dusenberrymw
'''
import random
import math
from researchproject.model.training import SigmoidActivationFunction


class Network():
    """A class for the overall network"""
    
    def __init__(self, num_inputs, activation_function,
                 num_hidden_neurons=None, num_output_neurons=1):
        """Constructor"""
        self.num_inputs = num_inputs
        if num_hidden_neurons is None:
            # Note: For now, take the mean of the number of inputs and outputs
            #    -THIS NEEDS TO BE CHANGED
            num_hidden_neurons = round((2/3) * num_inputs)
        self.hidden_layer = Layer(num_hidden_neurons, num_inputs, activation_function)
        self.output_layer = Layer(num_output_neurons, num_hidden_neurons, activation_function)
        self.layers = [self.hidden_layer, self.output_layer]
    
    def compute_network_output(self, inputs):
        """Compute output(s) of network given one entry (row) of data"""
#         if len(inputs) != self.num_inputs:
#             print("Error: Number of inputs(%d) previously defined(%d) and "
#                     "those given do not match" %(len(inputs), self.num_inputs))
#             traceback.print_stack()
#             exit(1)  
        outputs = []
        out = 0
        layers = self.layers
        neurons = []
        for layer in layers:
            outputs = []
            neurons = layer.neurons
            for neuron in neurons:
                # the first layer is the hidden neurons,
                #    so the inputs are those supplied to the
                #    network
                # then, for the output neurons, the inputs will
                #    be the outputs of the hidden neurons
                out = neuron.compute_output(inputs)
                outputs.append(out)
            # the inputs to the output neurons will be the outputs
            #    of the hidden neurons
            inputs = outputs[:]
        return outputs
    
    def calculate_error(self, inputs, target_outputs):
        """Determine the root mean square (RMS) error for the given multiple
        sets (rows) of input data against the associated target outputs
        
        RMS error = sqrt( (sum(residual^2)) / num_values )
        
        """
#        num_inputs = len(inputs)
        num_outputs = len(target_outputs)
#         if num_inputs != num_outputs:
#             print("Error: Number of input sets(%d) and target output " \
#                   "sets(%d) do not match" % (num_inputs,num_outputs))
#             exit(1)
        
        error = 0.0
        computed_output_set = []
        residual = 0
        num_values = num_outputs * len(target_outputs[0])
        compute_network_output = self.compute_network_output
        
        for input_set, target_output_set in zip(inputs, target_outputs):
            computed_output_set = compute_network_output(input_set)   
            
            for target_output_value, computed_output_value in \
                zip(target_output_set, computed_output_set):
                # error += math.fabs(target_outputs[i][j] - computed_outputs[j])
                residual = target_output_value - computed_output_value
                error += residual*residual  # square the residual value
        
        # average the error and take the square root
        return math.sqrt(error/num_values)


class Layer():
    """A class for layers in the network"""
    
    def __init__(self, num_neurons, num_inputs, activation_function):
        """Constructor"""
        self.neurons = []
        for _ in range(num_neurons):
            self.neurons.append(Neuron(num_inputs, activation_function))


class Neuron:
    """A class for neurons in the network"""
    
    def __init__(self, num_inputs, activation_function):
        """Constructor"""
        self.num_inputs = num_inputs
        self.activation_function = activation_function
        self.weights = [] #need a weight for each input to the neuron
        self.prev_weight_deltas = []
        for _ in range(num_inputs):
            self.weights.append(random.random())
            self.prev_weight_deltas.append(random.random())
        self.threshold = random.random()
        self.prev_threshold_delta = random.random()
        self.inputs = [] # the inputs coming from previous neurons
        self.local_output = 0.0 # the output leaving this neuron
        self.error_gradient = 0.0
    
    def compute_output(self, inputs):
        """Given a set of inputs from previous layer neuron,
        will compute the local output of the neuron
        """
        # keep track of what inputs were sent to this neuron
        self.inputs = inputs
        
        # multiply each input with the associated weight for that connection
        local_output = 0.0
        weights = self.weights  
        for input_value, weight_value in zip(inputs, weights):
            local_output += input_value * weight_value
        
        # then subtract the threshold value
        local_output += self.threshold * -1
        
        # finally, use the activation function to determine the output
        local_output = (self.activation_function.
            activate(local_output))
        
        # store outputs
        self.local_output = local_output
        
        return local_output


