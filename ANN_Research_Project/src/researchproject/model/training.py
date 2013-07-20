'''
Created on Jun 6, 2013

@author: dusenberrymw
'''
from math import exp, log, tanh, atanh, sqrt
from multiprocessing import Process, Queue, cpu_count


class Backpropagation(object):
    """Class for the Backpropagation type of training"""
    def __init__(self, learning_rate, momentum_coef):
        """ Constructor
        
        Learning rate = degree to which the weight and threshold values will
            be changed during each iteration of training
        Momentum = degree to which the previous learning will affect this 
            iteration's weight and threshold values. Effectively, it keeps 
            the weight values from oscillating during training
        
        """
        self.learning_rate = learning_rate
        self.momentum_coef = momentum_coef

    def train(self, network, inputs, target_outputs, iterations, min_error):
        """This trains the given network using the given multiple sets
        of input data (multiple rows) against the associated target outputs
        
        Start with the output layer and find the error gradient for each 
            output neuron.
        Then, for each neuron in the hidden layer(s), find the error gradient.
        Then, find the deltas and momentum of each neuron's weights and
            threshold values and add both to each weight and threshold.
        
        """
        num_input_sets = len(inputs)
        num_output_sets = len(target_outputs)
        if num_input_sets != num_output_sets:
            print("Error: Number of input sets(%d) and target output " \
                  "sets(%d) do not match" % (num_input_sets,num_output_sets))
            exit(1)
        
        # initialize variables for the following loops
        learning_rate = self.learning_rate
        momentum_coef = self.momentum_coef
        compute_network_output = network.compute_network_output # performance
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
                    for j, neuron in enumerate(layer.neurons):
                        prev_weight_deltas = neuron.prev_weight_deltas # REMOVE THIS
                        prev_threshold_delta = neuron.prev_threshold_delta # REMOVE THIS
                        
                        neuron_weights = neuron.weights
                        computed_output = neuron.local_output
                        
                        # The output layer neurons are treated slightly
                        #    different than the hidden neurons
                        if isOutputLayer:
                            # keep track of the error from this output neuron
                            #    for use in determining how well the network
                            #    is performing
                            # note: only need to do this for output neurons
                            residual = target_output_set[j] - computed_output
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
                                sum_value += gradient * weights[j]
                            error_gradient = (derivative(computed_output) * 
                                              sum_value)
                        
                        # now store the error gradient and the list of weights
                        #    for this neuron into these storage lists for the 
                        #    whole layer
                        this_layer_gradients.append(error_gradient)
                        this_layer_weights.append(neuron_weights)
                         
                        # Now, compute and add delta and momentum values
                        #     for each weight associated with this neuron
                        for k, (input_value, prev_weight_delta) in \
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
                            neuron_weights[k] += delta + momentum
                            # and update the previous weight delta value
                            prev_weight_deltas[k] = delta
                             
                        # now compute the delta and momentum for the threshold
                        #   value, by using a -1 as the threshold "input value"
                        delta = learning_rate * error_gradient * (-1)
                        momentum = momentum_coef * prev_threshold_delta
                        # then add these two values to the current threshold
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
        

class ResilientPropagation(object):
    """Class for the Resilient Propagation type of training"""
    def __init__(self):
        self.pos_step = 1.2
        self.neg_step = 0.5
        self.max_delta = 50.0
        self.min_delta = exp(-6)

    def train(self, network, inputs, target_outputs, iterations, min_error):
        """This trains the given network using the given multiple sets
        of input data (multiple rows) against the associated target outputs.
        
        Resilient propagation does not require a learning rate or a momentum
            value, as it does not rely on the value of the error gradients
            themselves, but rather on 
        
        Start with the output layer and find the error gradient for each 
            output neuron.
        Then, for each neuron in the hidden layer(s), find the error gradient.
        Then, for each neuron determine the change in sign between this error 
            gradient and the previous gradient.
        Then, for each weight and threshold, determine the delta, compute the
            weight change, and update the weights.
        
        """
        num_input_sets = len(inputs)
        num_output_sets = len(target_outputs)
        if num_input_sets != num_output_sets:
            print("Error: Number of input sets(%d) and target output " \
                  "sets(%d) do not match" % (num_input_sets,num_output_sets))
            exit(1)
            
        iteration_counter = 0
        error = network.calculate_error(inputs, target_outputs)
        while (iteration_counter < iterations) & (error > min_error):
            iteration_counter += 1
            self._compute_partial_gradients(network, inputs, target_outputs)
            self._update_weights(network)
            error = network.calculate_error(inputs, target_outputs)
        
    def _compute_partial_gradients(self, network, inputs, target_outputs):
        # find and store accumulated partial gradients for each 
        #    weight/threshold for all of the inputs
        #    -this is done as "Batch Training"
        # for each row of data
        for input_set, target_output_set in zip(inputs, target_outputs):
            # prime the network on this row of input data
            #    -this will cause local_output values to be
            #     set for each neuron
            network.compute_network_output(input_set) # see above
                       
            # For each neuron, starting with the output layer and moving
            #     backwards through the layers:
            #        -find and store the error gradient
            #        -determine the change in sign between this gradient
            #            and the previous one
            #        -compute the delta and weight change values
            prev_layer_gradients = []
            prev_layer_weights = []
            isOutputLayer = True
            for layer in reversed(network.layers): # iterate backwards
                derivative = layer.activation_function.derivative
                this_layer_gradients = [] # values from current layer
                this_layer_weights = []
                for j, neuron in enumerate(layer.neurons):
                    neuron_weights = neuron.weights
                    computed_output = neuron.local_output
                    # The output layer neurons are treated slightly
                    #    different than the hidden neurons
                    if isOutputLayer:
                        residual = target_output_set[j] - computed_output
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
                            sum_value += gradient * weights[j]
                        error_gradient = (derivative(computed_output) * 
                                          sum_value)
                    # now store the error gradient and the list of weights
                    #    for this neuron into these storage lists for the 
                    #    whole layer
                    this_layer_gradients.append(error_gradient)
                    this_layer_weights.append(neuron_weights)
                    # Now, compute, accumulate, and store the partial
                    #    derivative (partial gradient) for each weight
                    for k in range(len(neuron_weights)):
                        # need the portion of the gradient (partial 
                        #    derivative) for this weight connection
                        partial_gradient = (error_gradient * 
                                            neuron.inputs[k])
                        # then accumulate the stored partial gradient
                        neuron.partial_weight_gradients[k] += partial_gradient
                    # Now, compute, accumulate, and store the partial
                    #    derivative (partial gradient) for each threshold
                    #    -use -1 as the 'input' to the threshold
                    partial_gradient = error_gradient * -1
                    # then accumulate the stored partial gradient
                    neuron.partial_threshold_gradient += partial_gradient
                # store the gradients and weights from the current layer
                #    for the next layer (moving backwards)
                prev_layer_gradients = this_layer_gradients
                prev_layer_weights = this_layer_weights
                isOutputLayer = False
                
    def _update_weights(self, network):
        for layer in network.layers:
            for neuron in layer.neurons:
                prev_weight_deltas = neuron.prev_weight_deltas # REMOVE THIS
                prev_threshold_delta = neuron.prev_threshold_delta # REMOVE THIS
                # Now, compute the weight delta for each weight 
                #    associated with this neuron
                for k, delta in \
                        enumerate(prev_weight_deltas):
                    # need the portion of the gradient (partial 
                    #    derivative) for this weight connection
                    partial_gradient = neuron.partial_weight_gradients[k]
                    prev_partial_gradient = \
                        neuron.prev_partial_weight_gradients[k]
                    # now determine the sign change between this and
                    #    the previous gradient values
                    gradient_sign_change = \
                        sign(prev_partial_gradient*partial_gradient)
                    # determine the updated delta
                    if gradient_sign_change > 0: # no sign change
                        delta = min(delta * self.pos_step, 
                                    self.max_delta)
                    elif gradient_sign_change < 0: # sign change
                        delta = max(delta * self.neg_step,
                                    self.min_delta)
                        partial_gradient = 0
                    # now update the weight
                    neuron.weights[k] -= sign(partial_gradient)*delta
                    # and update the previous weight delta value
                    prev_weight_deltas[k] = delta
                    # and store the prev partial gradient
                    neuron.prev_partial_weight_gradients[k] = partial_gradient
                    neuron.partial_weight_gradients[k] = 0
                # Now compute the threshold delta
                #     -need the portion of the gradient (partial 
                #        derivative) for this threshold
                partial_gradient = neuron.partial_threshold_gradient
                prev_partial_gradient = \
                    neuron.prev_partial_threshold_gradient
                # determine the sign change between this and
                #    the previous gradient values
                gradient_sign_change = \
                    sign(prev_partial_gradient*partial_gradient)
                # determine the updated delta
                delta = prev_threshold_delta
                if gradient_sign_change > 0:
                    delta = min(delta * self.pos_step, self.max_delta)
                elif gradient_sign_change < 0:
                    delta = max(delta * self.neg_step, self.min_delta)
                    partial_gradient = 0
                # then update the threshold
                neuron.threshold -= sign(partial_gradient)*delta
                # and update the previous threshold delta value
                neuron.prev_threshold_delta = delta
                # and store the prev partial gradient
                neuron.prev_partial_threshold_gradient = partial_gradient
                neuron.partial_threshold_gradient = 0


class SigmoidActivationFunction(object):
    """Class for one of the possible activation functions used by the network"""
    def activate(self, input_value):
        """Run the input value through the sigmoid function
        This will only return values between 0 and 1
        """
#         try:
        # constant -1.0 can be changed to alter the steepness of the function
        return 1.0 / (1 + exp(-1.0*input_value)) # using math.exp
#         except OverflowError:
#             exit()
    
    def derivative(self, input_value):
        """Some training will require the derivative of the Sigmoid function"""
        return input_value * (1.0-input_value) + 0.1 # add 0.1 to fix flat spot
    
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
        return (1.0-input_value) * (1.0+input_value) + 0.1 # add 0.1 to fix flat spot
    
    def inverse(self, input_value):
        """This will produce the inverse of the tanh function, which is
        useful in determining the original value before activation
        """
        return atanh(input_value) # using math.atanh


def parallel_train(network, trainer, inputs, target_outputs, iterations, 
                   min_error, num_processes=None):
    """Train the given network using the given trainer in parallel using 
    multiple processes.
    
    This will ultimately change the weights in the given network
    
    """
    if num_processes is None:
        # Determine the number of processes
        num_processes = cpu_count()
    
    if num_processes > len(inputs):
        # there is not enough input data to split amongst the entire possible
        #    number of processes, so reduce the number of processes used
        num_processes = len(inputs)
    
    # Create the return queue to store the training networks that are returned
    #    from each process
    results_queue = Queue() # this is the multiprocessing queue

    # Train the training networks on a subset of the data
    #    -this is where the parallelization will occur
    #    -To get the training data subset for each process:
    #        -start with the nth element, where n is this processes number, and
    #            skip by x elements, where x is the total number of processes
    #    -Note: when this process is created, a copy of any object that is 
    #        accessed will be made, so no need to make a copy of the given 
    #        network first
    jobs = [Process(target=_parallel_train_worker, 
                    args=(network, trainer, 
                          inputs[process_num::num_processes],
                          target_outputs[process_num::num_processes], 
                          iterations, min_error, results_queue))
                    for process_num in range(num_processes-1)]
    # start the processes
    for job in jobs: job.start()
    
    # while those processes are running, perform this process's work as well
    #    -Note: this process counts as one of the processes, so set to the 
    #        last process number possible
    _parallel_train_worker(network, trainer, 
                           inputs[num_processes-1::num_processes], 
                           target_outputs[num_processes-1::num_processes], 
                           iterations, min_error, results_queue)
    
    # retrieve the trained networks as they come in
    #    -Note: this is necessary because the multiprocess Queue is actually
    #        a pipe, and has a maximum size limit.  Therefore, it will not work
    #        unless these are pulled from the other end
    #    -Note: the get() will, by default, wait until there is an item ready
    #        with no timeout
    training_networks = [results_queue.get() for _ in range(num_processes)]
    
    # now wait for the other processes to finish
    for job in jobs: job.join()
    
    # now average out the training networks into the master network
    #    by averaging the weights and threshold values
    for i, layer in enumerate(network.layers):
        for j, neuron in enumerate(layer.neurons):
            # average out weights
            for k in range(len(neuron.weights)):
                weight_sum = 0.0
                prev_weight_delta_sum = 0.0
                prev_partial_weight_gradients_sum = 0.0
                for trained_network in training_networks:
                    weight_sum += trained_network.layers[i].neurons[j].weights[k]
                    prev_weight_delta_sum += trained_network.layers[i].neurons[j].prev_weight_deltas[k]
                    prev_partial_weight_gradients_sum += trained_network.layers[i].neurons[j].prev_partial_weight_gradients[k]
                neuron.weights[k] = weight_sum/len(training_networks)
                neuron.prev_weight_deltas[k] = prev_weight_delta_sum/len(training_networks)
                neuron.prev_partial_weight_gradients[k] = prev_partial_weight_gradients_sum/len(training_networks)
            # average out thresholds
            threshold_sum = 0.0
            prev_threshold_delta_sum = 0.0
            prev_partial_threshold_gradient_sum = 0.0
            for trained_network in training_networks:
                    threshold_sum += trained_network.layers[i].neurons[j].threshold
                    prev_threshold_delta_sum += trained_network.layers[i].neurons[j].prev_threshold_delta
                    prev_partial_threshold_gradient_sum += trained_network.layers[i].neurons[j].prev_partial_threshold_gradient
            neuron.threshold = threshold_sum/len(training_networks)
            neuron.prev_threshold_delta = prev_threshold_delta_sum/len(training_networks)
            neuron.prev_partial_threshold_gradient = prev_partial_threshold_gradient_sum/len(training_networks)


def _parallel_train_worker(network, trainer, inputs, target_outputs, 
                           iterations, min_error, results_queue):
    """This private function is run by a parallel process to train the
    network on a given subset of the training data"""
    trainer.train(network, inputs, target_outputs, 
                  iterations, min_error)
    # return this trained network back to the main process by placing it on
    #    the queue
    results_queue.put(network)          


def sign(x):
    """Return the sign (-1, 0, 1) of x"""
    if x < 0.0:
        sign = -1
    elif x == 0.0:
        sign = 0
    elif x > 0.0:
        sign = 1
    return sign
    
    