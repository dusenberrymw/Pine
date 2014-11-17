'''
Created on Jun 6, 2013

@author: dusenberrymw
'''
import random


class Network(object):
    """A class for the overall network"""

    def __init__(self):
        """Constructor"""
        self.layers = []

    def forward(self, input_vector):
        """
        Given an input vector, forward propagate it through the network, 
            setting the input_vectors for all neurons, and return the output 
            vector of the network

        """
        for layer in self.layers:
            input_vector = layer.forward(input_vector)
        output_vector = input_vector
        return output_vector

    def backward(self, downstream_gradient_vector):
        """
        Given a gradient vector, calculate the gradients (error) for all 
            network parameters (weights/thresholds), store in each neuron, 
            and return the gradient vector for the inputs to the network

        """
        for layer in reversed(self.layers):
            downstream_gradient_vector = layer.backward(downstream_gradient_vector)
        upstream_gradient_vector = downstream_gradient_vector
        return upstream_gradient_vector

    def cost(self, target_output_vector, reg_lambda=0):
        """
        Cost J ("error") of the network, which is the summation of the cost
            vector of the output layer, given the last input vector forward
            propagated through the network

        reg_lambda is a regularization coefficient that is multiplied by the
            sum of each theta^2 in the entire network.  Regularization serves
            to push the theta values (weights) towards 0, thus limiting the
            chance of overfitting.

        """
        thetas_squared = 0;
        for layer in self.layers:
            for neuron in layer.neurons:
                thetas_squared += sum([w**2 for w in neuron.weights])
        cost = (sum(self.layers[-1].cost(target_output_vector))
                + reg_lambda/2 * thetas_squared)
        return cost

    def cost_gradient(self, target_output_vector):
        """
        Cost gradient vector (partial derivatives) of the network, which is 
            the cost gradient vector of the output layer, given the last input
            vector forward propagated through the network

        """
        cost_gradient_vector = self.layers[-1].cost_gradient(target_output_vector)
        return cost_gradient_vector

    def update_parameters(self, learning_rate, batch_size, reg_lambda=0):
        """
        Adjust the weights and threshold down the gradient to reduce error

        """
        for layer in self.layers:
            layer.update_parameters(learning_rate, batch_size, reg_lambda)

    def reset_gradients(self):
        """
        Set all parameter gradients to 0

        """
        for layer in self.layers:
            layer.reset_gradients()


class Layer(object):
    """A class for layers in the network"""

    def __init__(self):
        """Constructor"""
        self.neurons = []

    def forward(self, input_vector):
        """
        Given an input vector [from previous layer], compute the output vector
            of this layer of neurons in a forward pass

        """
        output_vector = [n.forward(input_vector) for n in self.neurons]
        return output_vector

    def backward(self, downstream_gradient_vector):
        """
        Given an error gradient vector from the downstream layer, calculate 
            the gradients (error) for this layer

        """
        gradient_vectors = [neuron.backward(downstream_gradient)
                            for neuron, downstream_gradient
                            in zip(self.neurons, downstream_gradient_vector)]
        # now reduce the vectors into one vector using element-wise summation,
        #   since each neuron in this layer shares the same input sources
        upstream_gradient_vector = gradient_vectors.pop()
        for gradient_vector in gradient_vectors:
            for i in range(len(gradient_vector)):
                upstream_gradient_vector[i] += gradient_vector[i]
        return upstream_gradient_vector

    def cost(self, target_output_vector):
        """
        Cost ("error") vector of this layer, given the last input vector 
            forward propagated through the layer

        """
        cost_vector = [neuron.cost(target_output)
                       for neuron, target_output
                       in zip(self.neurons, target_output_vector)]
        return cost_vector

    def cost_gradient(self, target_output_vector):
        """
        Cost gradient vector (partial derivatives) of this layer, given the 
            last input vector forward propagated through the layer

        """
        cost_gradient_vector = [neuron.cost_gradient(target_output)
                                for neuron, target_output
                                in zip(self.neurons, target_output_vector)]
        return cost_gradient_vector

    def update_parameters(self, learning_rate, batch_size, reg_lambda=0):
        """
        Adjust the weights and threshold down the gradient to reduce error

        """
        for neuron in self.neurons:
            neuron.update_parameters(learning_rate, batch_size, reg_lambda)

    def reset_gradients(self):
        """
        Set all parameter gradients to 0

        """
        for neuron in self.neurons:
            neuron.reset_gradients()


class Neuron(object):
    """A class for neurons in the network"""

    def __init__(self, num_inputs, activation_function):
        """Constructor"""
        self.input_vector = [0]*num_inputs # inputs coming from prev neurons
        self.output = 0.0 # the activation of this neuron
        self.activation_function = activation_function
        # need a weight for each input to the neuron
        self.weights = [random.uniform(-0.9,0.9) for _ in range(num_inputs)]
        self.threshold = random.uniform(-0.9,0.9)
        # gradients
        self.weight_gradients = [0]*num_inputs
        self.threshold_gradient = 0

    def forward(self, input_vector):
        """
        Given an input vector from previous layer neurons, compute the output 
            of the neuron in a forward pass

        Explanation:

            self.output = f(z), where z = w1x1 + w2x2 + ... + wnxn + thresh
                                and f() is the activation function

        """
        # keep track of what inputs were sent to this neuron
        self.input_vector = input_vector
        # multiply each input with the associated weight for that connection,
        #  then add the threshold value
        net_input = (sum([x*y for x, y in zip(input_vector, self.weights)])
                     + self.threshold)
        # finally, use the activation function to compute the output
        self.output = self.activation_function.activate(net_input)
        return self.output

    def backward(self, downstream_gradient):
        """
        Given an error gradient from the downstream layer, calculate the
            gradients (error) for each of the parameters (weights & threshold)
            and add to the existing corresponding gradients (for batch 
            purposes).

        Explanation:

            self.output = f(z), where z = w1x1 + w2x2 + ... + wnxn + thresh

            Therefore, the derivative of the output wrt weight i is:
            
                doutput/dwi = f'(z) * dz/dwi, where dz/dwi = xi
                
            The gradient ("downstream error") is passed in and multiplied by
                the derivative wrt to a weight to compute the gradient for 
                that weight.

        """
        chain_gradient = (downstream_gradient * 
                          self.activation_function.derivative(self.output))
        for i in range(len(self.weights)):
            self.weight_gradients[i] += chain_gradient * self.input_vector[i]
        self.threshold_gradient += chain_gradient#* 1, b/c thresh 'input' = 1
        input_gradients = [0]*len(self.input_vector)
        for i in range(len(self.input_vector)):
            input_gradients[i] = chain_gradient * self.weights[i]
        return input_gradients

    def cost(self, target_output):
        """
        Cost ("error") of this neuron, given the last input vector forward
            propagated through the neuron

        """
        cost = self.activation_function.cost(self.output, target_output)
        return cost

    def cost_gradient(self, target_output):
        """
        Cost gradient (partial derivative of cost) of this neuron, given the
            last input vector forward propagated through the neuron

        This will basically determine how much the hypothesis (output of the
            neuron) contributed to the cost ("error") of the neuron

        """
        cost_gradient = self.activation_function.cost_derivative(self.output, 
                                                                 target_output)
        return cost_gradient

    def update_parameters(self, learning_rate, batch_size, reg_lambda=0):
        """
        Update each neuron's weights and threshold value by subtracting the
            average gradient multiplied by the learning rate, alpha.  Subtract
            because the gradient will give direction of cost increase, and 
            we want to move in the opposite direction (gradient descent) in 
            order to lower the overall error (minimize the cost function, J).

        Also, use a lambda regularization term to penalize the weights
            further, which will push the weights towards 0.

        """
        for i in range(len(self.weights)):
            self.weights[i] -= (learning_rate/batch_size * 
                                (self.weight_gradients[i] + 
                                 reg_lambda*self.weights[i]))
        self.threshold -= learning_rate/batch_size * (self.threshold_gradient)

    def reset_gradients(self):
        """
        Set all parameter gradients to 0

        """
        self.weight_gradients = [0]*len(self.weight_gradients)
        self.threshold_gradient = 0

