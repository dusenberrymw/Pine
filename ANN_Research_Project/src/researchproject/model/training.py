'''
Created on Jun 6, 2013

@author: dusenberrymw
'''
import math


class Backpropagation(object):
    """Class for the Backpropagation type of training"""
    
    def __init__(self):
        """ Constructor"""

    def parallel_train(self, network, network_num, params, num_processes, results_queue):
        """This function is run by the parallel processes to each train their
        network on a subset of the training data"""
        # get the training data
        training_inputs_subset = \
                params['data'].training_inputs[network_num::num_processes]
        training_target_outputs_subset = \
                params['data'].training_target_outputs[network_num::num_processes]
        # now train
        self.train(network, training_inputs_subset, 
                      training_target_outputs_subset, 
                      params['learning_rate'], params['momentum_coef'],
                      params['iterations'], params['min_error'])
        # return this trained network back to the main process by placing it on
        #    the queue
        results_queue.put(network)

    def train(self, network, inputs, target_outputs, learning_rate=0.7, momentum_coef=0.9,
              iterations=1000, min_error=0.001):
        """This trains the given network using the given multiple
        sets of data (multiple rows) against the associated target outputs
        
        Start with output layer and find error gradient for each output neuron.
        Then, for each neuron in the hidden layer(s), find the error gradient.
        Then, find the deltas and momentum of each neuron's weights and threshold 
            values and add both to each weight and threshold.
        
        Learning rate = degree to which the weight and threshold values will
            be changed during each iteration of training
        Momentum = degree to which the previous learning will affect this 
            iteration's weight and threshold values. Effectively, it keeps 
            the weight values from oscillating during training
        
        """
        num_input_sets = len(inputs)
        num_output_sets = len(target_outputs)
        if num_input_sets != num_output_sets:
            print("Error: Number of input sets(%d) and target output " \
                  "sets(%d) do not match" % (num_input_sets,num_output_sets))
            exit(1)
        
        # initialize variables for the following loops for performance
#         compute_network_output = network.compute_network_output # for performance
        activate = network.activation_function.activate
        derivative = network.activation_function.derivative # for performance  
        num_values = num_output_sets * len(target_outputs[0])
        layers = network.layers
        
        # start learning
        iteration_counter = 0
        error = network.calculate_error(inputs, target_outputs)
        while (iteration_counter < iterations) & (error > min_error):
            iteration_counter += 1
            error = 0.0 # clear out the error
            # for each row of data
            for input_set, target_output_set in zip(inputs, target_outputs):
                # prime the network on this row of input data
                #    -this will cause local_output values to be
                #     set for each neuron
#                 compute_network_output(input_set) # see above
                these_inputs = input_set
                for layer in network.layers:
                    outputs = []
                    for neuron in layer.neurons:
                        # the first layer is the hidden neurons,
                        #    so the inputs are those supplied to the
                        #    network
                        # then, for the next layers, the inputs will
                        #    be the outputs of the previous layer
                        
                        # keep track of what inputs were sent to this neuron
                        neuron.inputs = these_inputs
                        
                        # multiply each input with the associated weight for that connection
                        local_output = 0.0  
                        for input_value, weight_value in zip(these_inputs, neuron.weights):
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
                    these_inputs = outputs
                
                                        
                # For each neuron, starting with the output layer and moving
                #     backwards through the layers:
                #        -find and store the error gradient
                #        -compute and add the delta and momentum values to the
                #            weight and threshold values
                gradients_and_weights = []
                isOutputLayer = True
                for layer in reversed(layers): # iterate backwards
                    new_gradients_and_weights = [] # values from current layer
                    for i, neuron in enumerate(layer.neurons):
                        prev_weight_deltas = neuron.prev_weight_deltas
                        neuron_weights = neuron.weights
                        computed_output = neuron.local_output
                        
                        # The output layer neurons are treated slightly
                        #    different than the hidden neurons
                        if isOutputLayer:
                            # keep track of the error from this output neuron
                            #    for use in determining how well the network
                            #    is performing
                            # note: only need to do this for output neurons
                            residual = target_output_set[i] - computed_output
                            error += residual*residual  # square the residual
                            
                            error_gradient = (derivative(computed_output) * 
                                              residual)
                        # for the hidden layer neurons
                        else:
                            # Need to sum the products of the error gradient of
                            #    a neuron in the next layer and the weight 
                            #    associated with the connection between this 
                            #    hidden layer neuron and that neuron.
                            # This will basically determine how much this 
                            #    neuron contributed to the error of the neuron 
                            #    it is connected to
                            sum_value = 0.0
                            for (gradient, weights) in gradients_and_weights:
                                sum_value += gradient * weights[i]
                            
                            error_gradient = (derivative(computed_output) * 
                                              sum_value)
                        
                        # now store the error gradient and the list of weights
                        #    for this neuron as a tuple
                        new_gradients_and_weights.append((error_gradient,
                                                      neuron_weights))
                        
                        # Now, compute and add delta and momentum values
                        #     for each weight associated with this neuron
                        for i, (input_value, prev_weight_delta) in \
                                enumerate(zip(neuron.inputs, 
                                              prev_weight_deltas)):
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
                            neuron_weights[i] += delta + momentum
                            # and update the previous weight delta value
                            prev_weight_deltas[i] = delta
                         
                         
                        # now compute the delta and momentum for the threshold 
                        #   value, by using a -1 as the threshold "input value"
                        delta = (learning_rate * error_gradient * (-1))
                        momentum = (momentum_coef * 
                                    neuron.prev_threshold_delta)
                           
                        # now add these two values to the current threshold
                        neuron.threshold += delta + momentum
                        # and update the previous threshold delta value
                        neuron.prev_threshold_delta = delta
                        
                    
                    # store the gradients and weights from the current layer
                    #    for the next layer (moving backwards)
                    gradients_and_weights = new_gradients_and_weights
                    isOutputLayer = False
                
                
            # now compute the new error before the next iteration of the
            #    while loop by averaging the error and take the square root
            error = math.sqrt(error/num_values)
        
        # this is after the while loop
        self.iterations = iteration_counter        
        

class SigmoidActivationFunction(object):
    """Class for one of the possible activation functions used by the network"""
    
    def activate(self, input_value):
        """Run the input value through the sigmoid function"""
        return 1.0 / (1 + math.exp(-1.0*input_value))
    
    def derivative(self, input_value):
        """Some training will require the derivative of the Sigmoid function"""
        return input_value * (1.0-input_value)
    
