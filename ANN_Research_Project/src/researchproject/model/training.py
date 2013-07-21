'''
Created on Jun 6, 2013

@author: dusenberrymw
'''
from math import exp, log, tanh, atanh
from multiprocessing import Process, Queue, cpu_count


class Backpropagation(object):
    """Class for the Backpropagation type of training"""
    def __init__(self, learning_rate, momentum_coef):
        """ Constructor
        
        Learning rate = degree to which the parameters (weight and threshold
            values) will be changed during each iteration of training
        Momentum = degree to which the previous learning will affect this 
            iteration's weight and threshold values. Effectively, it keeps 
            the weight values from oscillating during training
        
        """
        self.learning_rate = learning_rate
        self.momentum_coef = momentum_coef

    def train(self, network, inputs, target_outputs, iterations):
        """This trains the given network using the given vectors
        of input data (multiple rows) against the associated target outputs
        
        Essentially look forward to determine error, then look backward to
            change weights and threshold values (which are currently stored
            on the current node) on connections from nodes in previous
            (backwards) layer
        
        Start with the output layer and find the (lowercase) delta (partial 
            deriv of the cost function, J, with respect to the node's input), 
            and the gradient (partial deriv of the cost function, J, with 
            respect to one of the weights on the incoming inputs) for each 
            output neuron.
        Then, for each neuron in the hidden layer(s), find the delta and 
            gradient.
        Then, update each neuron's weights and threshold value by subtracting
            gradient multiplied by the learning rate, alpha.  Subtract because
            the gradient (and delta values since they are both partial derivs)
            will give direction of gradient AScent, and we want to move in the
            opposite direction in order to lower the overall error (minimize
            the cost function, J).
        
        """
        for iteration_counter in range(iterations):
            # for each row of data
            for input_vector, target_output_vector in zip(inputs, target_outputs):
                # prime the network on this row of input data
                #    -this will cause local_output (activation) values to be
                #     set for each neuron
                network.compute_network_output(input_vector)
                           
                # Note: next_layer_deltas is a vector of the single
                #    delta values for each node in the next 
                #    (forward) layer
                next_layer_deltas = []
                next_layer_weights = []
                isOutputLayer = True
                for layer in reversed(network.layers): # iterate backwards
                    derivative = layer.activation_function.derivative
                    this_layer_deltas = [] # values from current layer
                    this_layer_weights = []
                    for j, neuron in enumerate(layer.neurons):                        
                        # The output layer neurons are treated slightly
                        #    different than the hidden neurons
                        if isOutputLayer:
                            # simply subtract the target from the hypothesis
                            delta = neuron.local_output - target_output_vector[j]
                        else: # for the hidden layer neurons
                            # Need to sum the products of the delta of
                            #    a neuron in the next (forward) layer and the 
                            #    weight associated with the connection between 
                            #    this hidden layer neuron and that neuron.
                            # This will basically determine how much this 
                            #    neuron contributed to the error of the neuron 
                            #    it is connected to
                            # Note: next_layer_deltas is a vector of the single
                            #    delta values for each node in the next 
                            #    (forward) layer
                            sum_value = 0.0
                            for next_delta, weights in zip(next_layer_deltas, 
                                                         next_layer_weights):
                                sum_value +=  weights[j] * next_delta
                            delta = (derivative(neuron.local_output) * 
                                              sum_value) 
                        
                        # now store the delta and the list of weights
                        #    for this neuron into these storage lists for the 
                        #    whole layer
                        this_layer_deltas.append(delta)
                        this_layer_weights.append(neuron.weights)
                         
                        # Now, compute the gradient (partial deriv of cost 
                        #    fcn, J, w/ respect to parameter ij) for each 
                        #    weight_ij (parameter_ij) associated with 
                        #    this neuron
                        for ij, input_ij in enumerate(neuron.inputs):
                            # compute gradient (partial deriv of cost J w/
                            #    respect to parameter ij)
                            # Note: index ij means from a previous
                            #    layer node i to this layer node j
                            gradient_ij = input_ij * delta
                            # Note: Subtract in order to minimize error, since
                            #    partial derivs point in direction of gradient
                            #    AScent
                            neuron.weights[ij] -= self.learning_rate * gradient_ij
                             
                        # Now, compute the gradient (partial deriv of cost 
                        #    fcn, J, with respect to parameter ij) for the 
                        #    threshold value (parameter_0j), by using a "1" as
                        #    the threshold "input value"
                        # -Note: index 0j means from a previous
                        #    layer threshold node 0 (threshold always has 
                        #    index i=0) to this layer node j
                        #        -can also think of it as the threshold being
                        #            internal to this neuron
                        gradient_0j = (1) * delta
                        neuron.threshold -= self.learning_rate * gradient_0j
                        
                    # Once this layer is done, store the gradients and weights 
                    #    from the current layer for the next layer iteration 
                    #    (moving backwards)
                    next_layer_deltas = this_layer_deltas
                    next_layer_weights = this_layer_weights
                    isOutputLayer = False
        
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
        num_input_vectors = len(inputs)
        num_output_vectors = len(target_outputs)
        if num_input_vectors != num_output_vectors:
            print("Error: Number of input sets(%d) and target output " \
                  "sets(%d) do not match" % (num_input_vectors,num_output_vectors))
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
        for input_vector, target_output_vector in zip(inputs, target_outputs):
            # prime the network on this row of input data
            #    -this will cause local_output values to be
            #     set for each neuron
            network.compute_network_output(input_vector) # see above
                       
            # For each neuron, starting with the output layer and moving
            #     backwards through the layers:
            #        -find and store the error gradient
            #        -determine the change in sign between this gradient
            #            and the previous one
            #        -compute the delta and weight change values
            next_layer_deltas = []
            next_layer_weights = []
            isOutputLayer = True
            for layer in reversed(network.layers): # iterate backwards
                derivative = layer.activation_function.derivative
                this_layer_deltas = [] # values from current layer
                this_layer_weights = []
                for j, neuron in enumerate(layer.neurons):
                    neuron_weights = neuron.weights
                    computed_output = neuron.local_output
                    # The output layer neurons are treated slightly
                    #    different than the hidden neurons
                    if isOutputLayer:
                        residual = target_output_vector[j] - computed_output
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
                        for gradient, weights in zip(next_layer_deltas,
                                                     next_layer_weights):
                            sum_value += gradient * weights[j]
                        error_gradient = (derivative(computed_output) * 
                                          sum_value)
                    # now store the error gradient and the list of weights
                    #    for this neuron into these storage lists for the 
                    #    whole layer
                    this_layer_deltas.append(error_gradient)
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
                next_layer_deltas = this_layer_deltas
                next_layer_weights = this_layer_weights
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
    """The Logistic (Sigmoid) activation function, which is one of the
        possibilities that can be used by the network
        
    """
    def activate(self, input_value):
        """Run the input value through the sigmoid function
        This will only return values between 0 and 1
        """
        try:
            return 1.0 / (1 + exp(-1.0*input_value)) # using math.exp
        except OverflowError:
            # bound the numbers if there is an overflow
            if input_value < 0:
                return 0.00000000001  # logistic function goes to 0 for small x
            else:
                return 0.99999999999
    
    def derivative(self, input_value):
        """Some training will require the derivative of the Sigmoid function
        
        Given F(x) is the logistic (sigmoid) function, the derivative
            F'(x) = F(x) * (1 - F(x))
        
        """
        return input_value * (1.0-input_value) # can add 0.1 to fix flat spot
    
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
        return (1.0-input_value) * (1.0+input_value) # can add 0.1 to fix flat spot
    
    def inverse(self, input_value):
        """This will produce the inverse of the tanh function, which is
        useful in determining the original value before activation
        """
        return atanh(input_value) # using math.atanh


def parallel_train(network, trainer, inputs, target_outputs, iterations, 
                   num_processes=None):
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
                          iterations, results_queue))
                    for process_num in range(num_processes-1)]
    # start the processes
    for job in jobs: job.start()
    
    # while those processes are running, perform this process's work as well
    #    -Note: this process counts as one of the processes, so set to the 
    #        last process number possible
    _parallel_train_worker(network, trainer, 
                           inputs[num_processes-1::num_processes], 
                           target_outputs[num_processes-1::num_processes], 
                           iterations, results_queue)
    
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
    for l, layer in enumerate(network.layers):
        for j, neuron in enumerate(layer.neurons):
            # average out weights
            for ij in range(len(neuron.weights)):
                weight_sum = 0.0
                for trained_network in training_networks:
                    weight_sum += trained_network.layers[l].neurons[j].weights[ij]
                neuron.weights[ij] = weight_sum/len(training_networks)
            # average out thresholds
            threshold_sum = 0.0
            for trained_network in training_networks:
                    threshold_sum += trained_network.layers[l].neurons[j].threshold
            neuron.threshold = threshold_sum/len(training_networks)


def _parallel_train_worker(network, trainer, inputs, target_outputs, 
                           iterations, results_queue):
    """This private function is run by a parallel process to train the
    network on a given subset of the training data"""
    trainer.train(network, inputs, target_outputs, 
                  iterations)
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
    
    