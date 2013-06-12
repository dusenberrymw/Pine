'''
Created on Jun 6, 2013

@author: dusenberrymw
'''
import math


class Backpropagation():
    """Class for the Backpropagation type of training"""
    
    def __init__(self, learning_rate=0.7, momentum_coef=0.9):
        self.learning_rate = learning_rate
        self.momentum_coef = momentum_coef

    def train(self, inputs, target_outputs, network, activation_function,
              iterations=1000, min_error=0.01):
        """This trains the given network using the given multiple
        sets of data (multiple rows) against the associated target outputs
        
        Start with output layer -> find error gradient for each output neuron.
        Then, for each neuron in the hidden layer -> find the error gradient.
        Then, find the deltas and momentum of all weights and threshold 
            values and add to each weight and threshold.
        
        """
        num_inputs = len(inputs)
        num_outputs = len(target_outputs)
        if num_inputs != num_outputs:
            print("Error: Number of input sets(%d) and target output " \
                  "sets(%d) do not match" % (num_inputs,num_outputs))
            exit(1)
        
        # initialize variables for the following loops
        compute_network_output = network.compute_network_output # for performance
        calculate_error = network.calculate_error # for performance
        derivative = activation_function.derivative # for performance
        hidden_neurons = network.hidden_layer.neurons
        output_neurons = network.output_layer.neurons
        neuron = None
        iteration_counter = 0
        error = calculate_error(inputs, target_outputs)
        gradients = []
        error_gradient = 0.0
        out = 0
        i = 0
        sum_value = 0.0
        delta = 0.0
        momentum = 0.0
        learning_rate = self.learning_rate
        momentum_coef = self.momentum_coef
        

        while (iteration_counter < iterations) & (error > min_error):
            iteration_counter += 1
            # for each data entry
            for input_set, target_output_set in zip(inputs, target_outputs):
                # prime the network on this row of input data
                #    -this will cause local_output values to be
                #     set for each neuron
                compute_network_output(input_set) # see above
                
                # compute error gradients for output neuron(s)
                gradients = []
                for neuron, target_output_value in \
                    zip(output_neurons, target_output_set):
                    out = neuron.local_output
                    # the error_gradient defines the magnitude and direction 
                    #    of the error
                    error_gradient = ((target_output_value - out) *
                                             derivative(out))
                    neuron.error_gradient = error_gradient
                    gradients.append(error_gradient)
                
                # compute error gradients for hidden neurons
                i = 0 # counter for this hidden neuron
                for neuron in hidden_neurons:
                    out = neuron.local_output
                    # Need to sum the product of each output neuron's gradient
                    #    and the weight associated with the connection between
                    #    this hidden layer neuron and the output neuron
                    sum_value = 0.0
                    for output_neuron in output_neurons:
                        sum_value += (output_neuron.error_gradient * 
                                output_neuron.weights[i])
                    neuron.error_gradient = (derivative(out) * sum_value)
                    i += 1
                
                # compute deltas and momentum values for each weight and 
                #    threshold, and then add them to each weight and
                #    threshold value
                delta = 0.0
                momentum = 0.0
                for layer in network.layers:
                    for neuron in layer.neurons:
                        # compute the deltas and momentum values for weights
                        i = 0
                        input_set = neuron.inputs
                        prev_weight_deltas = neuron.prev_weight_deltas
                        for input_value, prev_weight_delta in \
                            zip(input_set, prev_weight_deltas):
                            # delta value for this weight is equal to the
                            #    product of the learning rate, the error
                            #    gradient, and the input to this neuron
                            #    on the connection associated with this weight
                            delta = (learning_rate * 
                                     neuron.error_gradient * input_value)
                            # momentum value for this weight is equal to the
                            #    product of the momentum coefficient and the
                            #    previous delta for this weight
                            # the momentum keeps the weight values from 
                            #    oscillating during training
                            momentum = momentum_coef * prev_weight_delta
                            
                            # now add these two values to the current weight
                            neuron.weights[i] += delta + momentum
                            # and update the previous delta value
                            neuron.prev_weight_deltas[i] = delta
                            i += 1
                        # now compute the delta and momentum for the threshold 
                        #    value
                        delta = (learning_rate * 
                                 neuron.error_gradient * (-1))
                        momentum = (momentum_coef * 
                                    neuron.prev_threshold_delta)
                        # now add these two values to the current weight
                        neuron.threshold += delta + momentum
                        # and update the previous delta value
                        neuron.prev_threshold_delta = delta
                
            # now compute the new error before the next iteration of the
            #    while loop
            error = calculate_error(inputs, target_outputs)
                
        self.iterations = iteration_counter # this is after the while loop
        
        

class SigmoidActivationFunction():
    """Class for one of the possible activation functions used by the network"""
    
    def activate(self, input_value):
        """Run the input value through the sigmoid function"""
        #print(input_value)
        return 1.0 / (1 + math.exp(-1.0*input_value))
    
    def derivative(self, input_value):
        """Some training will require the derivative of the Sigmoid function"""
        return input_value * (1.0-input_value)
    
