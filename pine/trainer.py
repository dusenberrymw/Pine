'''
Created on Oct 13, 2014

@author: dusenberrymw
'''
import random

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

    def train(self, network, training_examples, batch_size=1, passes=1):
        """
        Train the given network using the given example vectors
        of input data (index 1 for each example) against the associated
        target output(s) (index 0 for each example)

        """
        for _ in range(passes):
            random.shuffle(training_examples)
            for i, example in enumerate(training_examples):
                x = example[0]
                y = example[1]
                network.forward(x)
                network.backward(network.cost_gradient(y))
                if (i+1) % batch_size == 0:
                    # update every batch_size examples
                    network.update_parameters(batch_size, self.learning_rate)
                    network.reset_gradients()
            # update once more in case extra examples past last batch_update
            network.update_parameters(batch_size, self.learning_rate)
            network.reset_gradients()

