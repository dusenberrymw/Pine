'''
Created on Oct 13, 2014

@author: dusenberrymw
'''

class SGD(object):
    """
    Class for the Stochastic Gradient Descent (backpropagation) trainer
    
    """
    def __init__(self, learning_rate=0.01):
        """ Constructor

        Learning rate = degree to which the parameters (weight and threshold
            values) will be changed during each parameter update 

        """
        self.learning_rate = learning_rate

    def train(self, network, input_vector, target_output_vector):
        for x, y in zip(input_vector, target_output_vector):
            network.forward(x)
            network.backward(network.cost_gradient(y))
            network.update_parameters()
            network.reset_gradients()

