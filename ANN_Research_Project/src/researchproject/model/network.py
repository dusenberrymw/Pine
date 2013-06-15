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
        self.activation_function = activation_function
        if num_hidden_neurons is None:
            # General rules of thumb:
            #     -The number of hidden neurons should be between the size of
            #        the input layer and the size of the output layer.
            #     -The number of hidden neurons should be 2/3 the size of the 
            #        input layer, plus the size of the output layer.
            #     -The number of hidden neurons should be less than twice the 
            #        -size of the input layer.
            num_hidden_neurons = round((2/3) * num_inputs) + num_output_neurons
        self.hidden_layer = Layer(num_hidden_neurons, num_inputs)
        self.output_layer = Layer(num_output_neurons, num_hidden_neurons)
        self.layers = [self.hidden_layer, self.output_layer]
    
    def compute_network_output(self, inputs):
        """Compute output(s) of network given one entry (row) of data"""
#         if len(inputs) != self.num_inputs:
#             print("Error: Number of inputs(%d) previously defined(%d) and "
#                     "those given do not match" %(len(inputs), self.num_inputs))
#             traceback.print_stack()
#             exit(1)  
        outputs = []
        local_output = 0.0
        layers = self.layers
        neurons = []
        activate = self.activation_function.activate
        
        for layer in layers:
            outputs = []
            neurons = layer.neurons
            for neuron in neurons:
                # the first layer is the hidden neurons,
                #    so the inputs are those supplied to the
                #    network
                # then, for the next layers, the inputs will
                #    be the outputs of the previous layer
                
                # NOTE: for performance reasons, this method call
                #    is being eliminated, and is being inlined below
#                 local_output = neuron.compute_output(inputs, self.activation_function)
#                 outputs.append(local_output)

                # keep track of what inputs were sent to this neuron
                neuron.inputs = inputs
                
                # multiply each input with the associated weight for that connection
                local_output = 0.0
                weights = neuron.weights  
                for input_value, weight_value in zip(inputs, weights):
                    local_output += input_value * weight_value
                
                # then subtract the threshold value
                local_output += neuron.threshold * -1
                
                # finally, use the activation function to determine the output
                local_output = activate(local_output)
                
                # store outputs
                neuron.local_output = local_output
                outputs.append(local_output)
                
            # the inputs to the next layer will be the outputs
            #    of the previous layer
            inputs = outputs[:]
        return outputs
    
    def calculate_error(self, inputs, target_outputs):
        """Determine the root mean square (RMS) error for the given dataset
        (multiple rows) of input data against the associated target outputs
        
        RMS error is the square root of: the sum of the squared differences 
        between target outputs and actual outputs, divided by the total number
        of values
        
        error = sqrt( (sum(residual^2)) / num_values )
        
        """
        error = 0.0
        computed_output_set = []
        residual = 0
        compute_network_output = self.compute_network_output # for performance
        
        # for each row of data
        for input_set, target_output_set in zip(inputs, target_outputs):
            computed_output_set = compute_network_output(input_set)
            
            for target_output_value, computed_output_value in \
                    zip(target_output_set, computed_output_set):
                residual = target_output_value - computed_output_value
                error += residual*residual  # square the residual value
        
        # average the error and take the square root
        num_values = len(target_outputs) * len(target_outputs[0])
        return math.sqrt(error/num_values)


class Layer():
    """A class for layers in the network"""
    
    def __init__(self, num_neurons, num_inputs):
        """Constructor"""
        self.neurons = []
        for _ in range(num_neurons):
            self.neurons.append(Neuron(num_inputs))


class Neuron:
    """A class for neurons in the network"""
    
    def __init__(self, num_inputs):
        """Constructor"""
        self.inputs = [] # the inputs coming from previous neurons
        self.local_output = 0.0 # the output leaving this neuron
        self.error_gradient = 0.0
        self.weights = [] # need a weight for each input to the neuron
        self.prev_weight_deltas = []
        for _ in range(num_inputs):
            self.weights.append(random.random())
            self.prev_weight_deltas.append(random.random())
        self.threshold = random.random()
        self.prev_threshold_delta = random.random()
    
    def compute_output(self, inputs, activation_function):
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
        local_output = (activation_function.
            activate(local_output))
        
        # store outputs
        self.local_output = local_output
        
        return local_output


