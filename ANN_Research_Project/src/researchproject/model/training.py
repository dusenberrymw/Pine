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
        if len(inputs) != len(target_outputs):
            print("Error: Number of input sets(%d) and target output " \
                  "sets(%d) do not match" % (len(inputs),len(target_outputs)))
            exit(1)
        
        # initialize variables for the following loops
        error = network.calculate_error(inputs, target_outputs)
        neuron = None
        gradients = []
        sum_value = 0.0
        iteration_counter = 0
        
        while (iteration_counter < iterations) & (error > min_error):
            iteration_counter += 1
            # for each data entry
            for i in range(len(inputs)):
                # prime the network on this row of input data
                #    -this will cause local_output values to be
                #     set for each neuron
                network.compute_network_output(inputs[i])
                
                # compute error gradients for output neurons
                gradients = []
                for j in range(len(network.output_layer.neurons)):
                    # may have multiple output neurons
                    neuron = network.output_layer.neurons[j]
                    out = neuron.local_output
                    # the error_gradient defines the magnitude and direction 
                    #    of the error
                    neuron.error_gradient = ((target_outputs[i][j] - out) *
                                             activation_function.derivative(out))
                    gradients.append(neuron.error_gradient)
                # compute error gradients for hidden neurons
                for j in range(len(network.hidden_layer.neurons)):
                    neuron = network.hidden_layer.neurons[j]
                    out = neuron.local_output
                    # Need to sum the product of each output neuron's gradient
                    #    and the weight associated with the connection between
                    #    this hidden layer neuron and the output neuron
                    sum_value = 0.0
                    for output_neuron in network.output_layer.neurons:
                        sum_value += (output_neuron.error_gradient * 
                                output_neuron.weights[j])
                    neuron.error_gradient = (activation_function.derivative(out) * 
                                             sum_value)
                
                # compute deltas and momentum values for each weight and 
                #    threshold, and then add them to each weight and
                #    threshold value
                delta = 0.0
                momentum = 0.0
                for layer in network.layers:
                    for neuron in layer.neurons:
                        # deltas and momentum values for weights
                        for i in range(len(neuron.weights)):
                            # delta value for this weight is equal to the
                            #    product of the learning rate, the error
                            #    gradient, and the input to this neuron
                            #    on the connection associated with this weight
                            delta = (self.learning_rate * 
                                     neuron.error_gradient * neuron.inputs[i])
                            # momentum value for this weight is equal to the
                            #    product of the momentum coefficient and the
                            #    previous delta for this weight
                            # the momentum keeps the weight values from 
                            #    oscillating during training
                            momentum = (self.momentum_coef * 
                                        neuron.prev_weight_deltas[i])
                            
                            # now add these two values to the current weight
                            neuron.weights[i] += delta + momentum
                            # and update the previous delta value
                            neuron.prev_weight_deltas[i] = delta
                        # now compute the delta and momentum for the threshold 
                        #    value
                        delta = (self.learning_rate * 
                                 neuron.error_gradient * (-1))
                        momentum = (self.momentum_coef * 
                                    neuron.prev_threshold_delta)
                        # now add these two values to the current weight
                        neuron.threshold += delta + momentum
                        # and update the previous delta value
                        neuron.prev_threshold_delta = delta
                
                # now compute the new error
                error = network.calculate_error(inputs, target_outputs)
                #print("Iteration: %d, Error: %f" % (iteration_counter, error))                
        
        

class SigmoidActivationFunction():
    """Class for one of the possible activation functions used by the network"""
    
    def activate(self, input_value):
        """Run the input value through the sigmoid function"""
        #print(input_value)
        return 1.0 / (1 + math.exp(-1.0*input_value))
    
    def derivative(self, input_value):
        """Some training will require the derivative of the Sigmoid function"""
        return input_value * (1.0-input_value)