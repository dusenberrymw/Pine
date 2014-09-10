'''
Created on Jun 6, 2013

@author: dusenberrymw
'''
import math
import random


class Network(object):
    """A class for the overall network"""

    def __init__(self):
        """Constructor"""
        self.layers = []

    def forward(self, input_vector):
        """
        Given an input vector, compute the output vector of the network

        """
        output_vector = input_vector
        for layer in self.layers:
            output_vector = layer.forward(output_vector)
        return output_vector


class Layer(object):
    """A class for layers in the network"""

    def __init__(self, num_neurons, num_inputs, activation_function):
        """Constructor"""
        self.activation_function = activation_function
        self.neurons = [Neuron(num_inputs) for _ in range(num_neurons)]

    def forward(self, input_vector):
        """
        Given an input vector from previous layer, compute the output vector of
            this layer of neurons in a forward pass

        """
        output_vector = [n.forward(input_vector, self.activation_function)
                         for n in self.neurons]
        return output_vector


class Neuron(object):
    """A class for neurons in the network"""

    def __init__(self, num_inputs):
        """Constructor"""
        self.input_vector = [] # the inputs coming from previous neurons
        self.output = 0.0 # the activation of this neuron
        # need a weight for each input to the neuron
        self.weights = [random.uniform(-0.9,0.9) for _ in range(num_inputs)]
        self.threshold = random.uniform(-0.9,0.9)

    def forward(self, input_vector, activation_function):
        """
        Given an input vector from previous layer neurons, compute the output of
            the neuron in a forward pass

        """
        # keep track of what inputs were sent to this neuron
        self.input_vector = input_vector
        # multiply each input with the associated weight for that connection,
        #  then add the threshold value
        local_output = (sum([x * y for x, y in zip(input_vector, self.weights)])
                        + self.threshold)
        # finally, use the activation function to determine the output
        self.output = activation_function.activate(local_output)
        return self.output
