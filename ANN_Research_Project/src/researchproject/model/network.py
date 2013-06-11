'''
Created on Jun 6, 2013

@author: dusenberrymw
'''
import random
import math
import traceback
import sys
from researchproject.model.training import SigmoidActivationFunction


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
        if len(inputs) != len(self.weights):
            print("Error: Number of inputs(%d) and weights(%d) do not " \
                   "match" % (len(inputs),len(self.weights)))
            print(inputs)
            exit(1)
        
        # keep track of what inputs were sent to this neuron
        self.inputs = inputs
        
        # multiply each input with the associated weight for that connection
        self.local_output = 0.0 
        for i in range(len(inputs)):
            self.local_output += inputs[i] * self.weights[i]
        # then subtract the threshold value
        self.local_output += self.threshold * -1
        # finally, use the activation function to determine the output
        self.local_output = self.activation_function.\
            activate(self.local_output)
        return self.local_output
    
    def debug_print(self):
        print("Hi, I'm a neuron" + str(self.weights))


class Layer():
    """A class for layers in the network"""
    
    def __init__(self, num_neurons, num_inputs, activation_function):
        """Constructor"""
        self.neurons = []
        for _ in range(num_neurons):
            self.neurons.append(Neuron(num_inputs, activation_function))
    
    def debug_print(self):
        print("Hi, I'm a Layer")
        for i in range(len(self.neurons)):
            print("Neuron #" + str(i))
            self.neurons[i].debug_print()


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
        if len(inputs) != self.num_inputs:
            print("Error: Number of inputs(%d) previously defined(%d) and " \
                    "those given do not match" %(len(inputs), self.num_inputs))
            traceback.print_stack()
            exit(1)
            
        outputs = None
        out = 0
        for layer in self.layers:
            outputs = []
            for neuron in layer.neurons:
                out = neuron.compute_output(inputs)
                outputs.append(out)
            inputs = outputs[:]
        return outputs
    
    def calculate_error(self, inputs, target_outputs):
        """Determine the error rate for the given multiple
        sets of data (multiple rows) against the associated target outputs"""
        if len(inputs) != len(target_outputs):
            print("Error: Number of input sets(%d) and target output " \
                  "sets(%d) do not match" % (len(inputs),len(target_outputs)))
            exit(1)
        
        error = 0.0
        computed_outputs = []
        for i in range(len(inputs)):
            computed_outputs = self.compute_network_output(inputs[i])
            for j in range(len(computed_outputs)):
                error += math.fabs(target_outputs[i][j] - computed_outputs[j])
        
        return error     
    
    def debug_print(self):
        print("Hi, I'm a Network")
        for i in range(self.num_inputs):
            print("Input #" + str(i))
        print()
        for i in range(len(self.layers)):
            print("Layer #" + str(i))
            self.layers[i].debug_print()
            print()


