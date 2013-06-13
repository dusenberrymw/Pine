'''
Created on Jun 6, 2013

@author: dusenberrymw
'''
import math


class Backpropagation():
    """Class for the Backpropagation type of training"""
    
    def __init__(self, learning_rate=0.7, momentum_coef=0.9):
        """ Constructor
        
        Learning rate = degree to which the weight and threshold values will
            be changed during each iteration of training
        Momentum = degree to which the previous learning will affect this 
            iteration's weight and threshold values. Effectively, it keeps 
            the weight values from oscillating during training
        
        """
        # Note: these values will need to be adjusted by trial and error
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
        num_input_sets = len(inputs)
        num_output_sets = len(target_outputs)
        if num_input_sets != num_output_sets:
            print("Error: Number of input sets(%d) and target output " \
                  "sets(%d) do not match" % (num_input_sets,num_output_sets))
            exit(1)
        
        # initialize variables for the following loops
        compute_network_output = network.compute_network_output # for performance
        derivative = activation_function.derivative # for performance
        hidden_neurons = network.hidden_layer.neurons
        output_neurons = network.output_layer.neurons
        neuron = None
        iteration_counter = 0
        computed_output_value = 0
        error = network.calculate_error(inputs, target_outputs)
        residual = 0.0
        num_values = num_output_sets * len(target_outputs[0])
        error_gradient = 0.0
        gradients_and_weights = []
        sum_value = 0.0
        delta = 0.0
        momentum = 0.0
        learning_rate = self.learning_rate
        momentum_coef = self.momentum_coef
        

        while (iteration_counter < iterations) & (error > min_error):
            iteration_counter += 1
            error = 0.0 # clear out the error
            # for each row of data
            for input_set, target_output_set in zip(inputs, target_outputs):
                # prime the network on this row of input data
                #    -this will cause local_output values to be
                #     set for each neuron
                compute_network_output(input_set) # see above
                
                # compute error gradients for output neuron(s)
                # also keep track of total network error
                #     Note: only need to do this for output neurons
                gradients_and_weights = []
                for neuron, target_output_value in \
                        zip(output_neurons, target_output_set):
                    computed_output_value = neuron.local_output
                    
                    # keep track of the error from this output neuron for use
                    #    in determining how well the network is performing
                    # note: only need to do this for output neurons
                    residual = target_output_value - computed_output_value
                    error += residual*residual  # square the residual value
                    
                    # the error_gradient defines the magnitude and direction 
                    #    of the error for use in learning
                    error_gradient = residual * derivative(computed_output_value)
                    neuron.error_gradient = error_gradient
                    
                    # now store the error gradient and the list of weights for
                    #    this output neuron as a tuple
                    gradients_and_weights.append((error_gradient, neuron.weights))
                
                # compute error gradients for hidden neurons
                for i, neuron in enumerate(hidden_neurons):
                    computed_output_value = neuron.local_output
                    
                    # Need to sum the product of each output neuron's gradient
                    #    and the weight associated with the connection between
                    #    this hidden layer neuron and the output neuron
                    # Note: this information was stored in the previous loop
                    #    in a tuple, with the output neuron's gradient first,
                    #    followed by the list of weights for that neuron
                    sum_value = 0.0
                    for gradient_and_weights in gradients_and_weights:
                        sum_value += (gradient_and_weights[0] * 
                            gradient_and_weights[1][i])
                    neuron.error_gradient = (derivative(computed_output_value) * sum_value)
                
                # compute deltas and momentum values for each weight and 
                #    threshold, and then add them to each weight and
                #    threshold value
                delta = 0.0
                momentum = 0.0
                for layer in network.layers:
                    neurons = layer.neurons
                    for neuron in neurons:
                        input_set = neuron.inputs
                        error_gradient = neuron.error_gradient
                        prev_weight_deltas = neuron.prev_weight_deltas
                        
                        # compute the deltas and momentum values for each 
                        #    weight
                        for i, (input_value, prev_weight_delta) in \
                                enumerate(zip(input_set, prev_weight_deltas)):
                            # delta value for this weight is equal to the
                            #    product of the learning rate, the error
                            #    gradient, and the input to this neuron
                            #    on the connection associated with this weight
                            delta = (learning_rate * error_gradient * 
                                     input_value)
                            # momentum value for this weight is equal to the
                            #    product of the momentum coefficient and the
                            #    previous delta for this weight
                            # the momentum keeps the weight values from 
                            #    oscillating during training
                            momentum = momentum_coef * prev_weight_delta
                            
                            # now add these two values to the current weight
                            neuron.weights[i] += delta + momentum
                            # and update the previous weight delta value
                            neuron.prev_weight_deltas[i] = delta

                        # now compute the delta and momentum for the threshold 
                        #    value
                        delta = (learning_rate * error_gradient * (-1))
                        momentum = (momentum_coef * 
                                    neuron.prev_threshold_delta)
                        
                        # now add these two values to the current threshold
                        neuron.threshold += delta + momentum
                        # and update the previous threshold delta value
                        neuron.prev_threshold_delta = delta
                
            # now compute the new error before the next iteration of the
            #    while loop by averaging the error and take the square root
            error = math.sqrt(error/num_values)
        
        # this is after the while loop
        self.iterations = iteration_counter
        
        

class SigmoidActivationFunction():
    """Class for one of the possible activation functions used by the network"""
    
    def activate(self, input_value):
        """Run the input value through the sigmoid function"""
        return 1.0 / (1 + math.exp(-1.0*input_value))
    
    def derivative(self, input_value):
        """Some training will require the derivative of the Sigmoid function"""
        return input_value * (1.0-input_value)
    
