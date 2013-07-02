'''
Created on Jun 6, 2013

@author: dusenberrymw
'''
from math import exp, log, tanh, atanh, sqrt
from multiprocessing import Process, Queue
import copy


class Backpropagation(object):
    """Class for the Backpropagation type of training"""
    def __init__(self):
        self.iterations = None
        

    def parallel_train(self, network, params, process_num, num_processes, results_queue):
        """This function is run by parallel processes to each train their
        network on a subset of the training data"""
        # get the training data
        #    -start with the nth element, where n is this processes number, and
        #        skip by x elements, where x is the total number of processes
        training_inputs_subset = \
                params['data'].training_inputs[process_num::num_processes]
        training_target_outputs_subset = \
                params['data'].training_target_outputs[process_num::num_processes]
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
        
        # initialize variables for the following loops
        compute_network_output = network.compute_network_output # for performance 
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
                compute_network_output(input_set) # see above
                           
                # For each neuron, starting with the output layer and moving
                #     backwards through the layers:
                #        -find and store the error gradient
                #        -compute and add the delta and momentum values to the
                #            weight and threshold values
                prev_layer_gradients = []
                prev_layer_weights = []
                isOutputLayer = True
                for layer in reversed(layers): # iterate backwards
                    derivative = layer.activation_function.derivative
                    this_layer_gradients = [] # values from current layer
                    this_layer_weights = []
                    for i, neuron in enumerate(layer.neurons):
                        prev_weight_deltas = neuron.prev_weight_deltas # REMOVE THIS
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
                            #    a neuron in the next (forward) layer and the 
                            #    weight associated with the connection between 
                            #    this hidden layer neuron and that neuron.
                            # This will basically determine how much this 
                            #    neuron contributed to the error of the neuron 
                            #    it is connected to
                            sum_value = 0.0
                            for gradient, weights in zip(prev_layer_gradients, 
                                                         prev_layer_weights):
                                sum_value += gradient * weights[i]
                            error_gradient = (derivative(computed_output) * 
                                              sum_value)
                        
                        # now store the error gradient and the list of weights
                        #    for this neuron into this storage list for the 
                        #    whole layer
                        this_layer_gradients.append(error_gradient)
                        this_layer_weights.append(neuron_weights)
                         
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
                    prev_layer_gradients = this_layer_gradients
                    prev_layer_weights = this_layer_weights
                    isOutputLayer = False
                
            # now compute the new error before the next iteration of the
            #    while loop by averaging the error and take the square root
            error = sqrt(error/num_values) # using math.sqrt
        
        # Note: this is after the while loop
        self.iterations = iteration_counter        
        

class SigmoidActivationFunction(object):
    """Class for one of the possible activation functions used by the network"""
    def activate(self, input_value):
        """Run the input value through the sigmoid function
        This will only return values between 0 and 1
        """
        return 1.0 / (1 + exp(-1.0*input_value)) # using math.exp
    
    def derivative(self, input_value):
        """Some training will require the derivative of the Sigmoid function"""
        return input_value * (1.0-input_value)
    
    def inverse(self, input_value):
        """This will produce the inverse of the sigmoid function, which is
        useful in determining the original value before activation
        """
        return log(input_value/(1-input_value)) # using math.log
    

class TanhActivationFunction(object):
    """Class for one of the possible activation functions used by the network"""
    def activate(self, input_value):
        """Run the input value through the tanh function"""
        return tanh(input_value) #using math.tanh
    
    def derivative(self, input_value):
        """Some training will require the derivative of the tanh function"""
        return (1.0-input_value) * (1.0+input_value)
    
    def inverse(self, input_value):
        """This will produce the inverse of the tahn function, which is
        useful in determining the original value before activation
        """
        return atanh(input_value) # using math.atanh


def parallel_training(master_network, trainer, params):
    """Train the given network over multiple processes using the given trainer
    
    This will ultimately change the weights in the given master network
    
    """
    # Determine the number of processes
    num_processes = params['num_processes']
    
    # Create the return queue to store the training networks that are returned
    #    from each process
    results_queue = Queue() # this is the multiprocessing queue

    # Train the training networks on a subset of the data
    #    -this is where the parallelization will occur
    #    -Note: when this process is created, a copy of all objects will be
    #        made, so no need to make a copy of the master network first
    #    -Note: this process counts as one of the processes, so be sure to
    #        have this one do work as well, set as the last process number
    jobs = [Process(target=trainer.parallel_train, 
                    args=(master_network, params, process_num, num_processes, 
                          results_queue))
                    for process_num in range(num_processes-1)]
    # start the other processes
    for job in jobs: job.start()
    
    # while those processes are running, perform this process's work as well
    #    -make a copy of the master network because this process is the one
    #    initially created the network
    trainer.parallel_train(copy.deepcopy(master_network), params, 
                           num_processes-1, num_processes, results_queue)
    
    # retrieve the trained networks as they come in
    #    Note: this is necessary because the multiprocess Queue is actually
    #        a pipe, and has a maximum size limit.  Therefore, it not work 
    #        unless these are pulled from the other end
    #    Note: the get() will, by default, wait until there is an item ready
    #        with no timeout
    training_networks = [results_queue.get() for _ in range(num_processes)]
    
    # now wait for the other processes to finish
    for job in jobs: job.join()
    
    # now average out the training networks into the master network
    #    by averaging the weights and threshold values
    for i, layer in enumerate(master_network.layers):
        for j, neuron in enumerate(layer.neurons):
            # average out weights
            for k in range(len(neuron.weights)):
                weight_sum = 0.0
                prev_weight_delta_sum = 0.0
                for network in training_networks:
                    weight_sum += network.layers[i].neurons[j].weights[k]
                    prev_weight_delta_sum += network.layers[i].neurons[j].prev_weight_deltas[k]
                neuron.weights[k] = weight_sum/num_processes
                neuron.prev_weight_deltas[k] = prev_weight_delta_sum/num_processes
            # average out thresholds
            threshold_sum = 0.0
            prev_threshold_delta_sum = 0.0
            for network in training_networks:
                    threshold_sum += network.layers[i].neurons[j].threshold
                    prev_threshold_delta_sum += network.layers[i].neurons[j].prev_threshold_delta
            neuron.threshold = threshold_sum/num_processes
            neuron.prev_threshold_delta = prev_threshold_delta_sum/num_processes
            

