'''
Created on Jun 6, 2013

@author: dusenberrymw
'''
import math
import random


class Network(object):
    """A class for the overall network"""

    def __init__(self, num_inputs, num_neurons_list, activation_functions):
        """Constructor

        'num_neurons_list' = list containing the number of neurons in each
                                layer after the input layer, since we do
                                not make objects for the inputs
        'activation_functions' = list containing the activation functions for
                                    each layer

        """
        self.layers = []
        for num_neurons, activation_function in \
                zip(num_neurons_list, activation_functions):
            if num_neurons is None:
                # Then automatically determine number of neurons
                # -General rules of thumb for number of nodes:
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

    def compute_network_output(self, input_vector):
        """Compute output(s) of network given one entry vector (row) of data"""
        for layer in self.layers:
            activate = layer.activation_function.activate
            output_vector = []
            # the first layer is the hidden neurons,
            #    so the inputs are those supplied to the
            #    network
            # then, for the next layers, the inputs will
            #    be the outputs of the previous layer
            for neuron in layer.neurons:
                # keep track of what inputs were sent to this neuron
                neuron.inputs = input_vector
                # multiply each input with the associated weight for that
                #    connection
                # Note: local_output is the "activation" of the neuron
                local_output = 0.0
                for input_value, weight_value in zip(input_vector, neuron.weights):
                    local_output += input_value * weight_value
                # then add the threshold value
                local_output += neuron.threshold
                # finally, use the activation function to determine the
                #    activated output
                local_output = activate(local_output)
                # store activated output
                neuron.local_output = local_output
                output_vector.append(local_output)
            # the inputs to the next layer will be the outputs
            #    of the previous layer
            input_vector = output_vector
        return output_vector  # this will be the vector of output layer activations


    def cost_J(self, example):
        """Determine the overall cost J(theta) for the network

        The overal cost, J(theta), is the overall "error" of the network
            with respect to parameter (weight) theta, and is equal to the
            sum of the cost functions for each node in the output layer,
            evaluated at the given training example

        examples are in the format: [[target_vector], [input_vector]]

        """
        target_output_vector = example[0]
        input_vector = example[1]
        output_layer = self.layers[-1]
        cost_func = output_layer.activation_function.cost
        hypothesis_vector = self.compute_network_output(input_vector)
        return sum([cost_func(hypothesis_vector[i], target_output_vector[i]) for i in range(len(output_layer.neurons))])


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
        self.local_output = 0.0 # the activation of this neuron
        # need a weight for each input to the neuron
        self.weights = [random.uniform(-0.9,0.9) for _ in range(num_inputs)]
        self.threshold = random.uniform(-0.9,0.9)

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
        # then add the threshold value
        local_output += self.threshold
        # finally, use the activation function to determine the output
        local_output = (activation_function.
            activate(local_output))
        # store outputs
        self.local_output = local_output
        return local_output
