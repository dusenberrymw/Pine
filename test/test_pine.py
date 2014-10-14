#! /usr/bin/env python3
'''
Created on Sept 9, 2014

@author: dusenberrymw
'''
import math
import os
import sys
import unittest

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

# import pine.data
import pine.activation
import pine.network
import pine.training
import pine.util

# network.py
class TestNetwork(unittest.TestCase):
    """Testing for network.py"""
    def setUp(self):
        self.act_func = pine.activation.Logistic()
        self.input_vector = [5,6,7]

        self.neuron = pine.network.Neuron(3, self.act_func)
        self.neuron.weights = [1,-2,3]
        self.neuron.threshold = 4

        local_output = sum([x*y for x,y in zip(self.input_vector, self.neuron.weights)]) + self.neuron.threshold
        self.output = 1.0 / (1 + math.exp(-1.0*local_output)) #0.99999999999

        self.layer = pine.network.Layer()
        self.layer.neurons = [self.neuron, self.neuron]

    def test_neuron_forward(self):
        self.assertEqual(self.neuron.forward(self.input_vector), self.output)

    def test_layer_forward(self):
        self.assertEqual(self.layer.forward(self.input_vector),
                         [self.output, self.output])

    def test_network_forward(self):
        network = pine.network.Network()
        network.layers.append(self.layer)
        new_neuron = pine.network.Neuron(2,self.act_func)
        new_neuron.weights = [1,-2]
        new_neuron.threshold = 4
        new_layer = pine.network.Layer()
        new_layer.neurons = [self.neuron]
        network.layers.append(new_layer)
        local_output = sum([x*y for x,y in zip([self.output, self.output], self.neuron.weights)]) + self.neuron.threshold
        out = [1.0 / (1 + math.exp(-1.0*local_output))]
        self.assertEqual(network.forward(self.input_vector), out) #0.9525741275104728

    def test_neuron_backward(self):
        self.neuron.forward(self.input_vector)
        self.neuron.output = 2
        down_gradient = 3.2
        chain_gradient = down_gradient * (2*(1-2))
        weight_gradients = [chain_gradient*x for x in self.input_vector]
        thresh_gradient = chain_gradient * 1
        input_gradients = [chain_gradient*x for x in self.neuron.weights]
        computed_gradients = self.neuron.backward(down_gradient)
        self.assertEqual(self.neuron.weight_gradients, weight_gradients)
        self.assertEqual(computed_gradients, input_gradients)
        self.assertEqual(self.neuron.threshold_gradient, thresh_gradient)

    def test_gradients(self):
        layout = [3,5,2]
        network = pine.util.create_network(layout, ['logistic']*2)
        input_vector = [-2.3,3.1,-5.8]
        target_output_vector = [0.4,1]
        network.forward(input_vector)
        cost_gradient_vec = network.cost_gradient(target_output_vector)
        network.backward(cost_gradient_vec)

        for layer in network.layers:
            for neuron in layer.neurons:
                # weight gradients check:
                for i in range(len(neuron.weights)):
                    epsilon = 0.0001
                    old_theta = neuron.weights[i]
                    neuron.weights[i] = neuron.weights[i] + epsilon
                    network.forward(input_vector)
                    J1 = network.cost(target_output_vector)
                    neuron.weights[i] = old_theta - epsilon
                    network.forward(input_vector)
                    J2 = network.cost(target_output_vector)
                    estimated_gradient = (J1 - J2) / (2*epsilon)
                    diff = abs(neuron.weight_gradients[i] - estimated_gradient)
                    assert diff < 0.0001, "w difference: {}".format(diff)
                    # print("w difference: {}".format(diff))
                    # print("weight_gradient[i]: {}".format(neuron.weight_gradients[i]))
                    # print("estimated_gradient: {}".format(estimated_gradient))
                    neuron.weights[i] = old_theta

                # threshold gradient check:
                epsilon = 0.0001
                old_theta = neuron.threshold
                neuron.threshold = neuron.threshold + epsilon
                network.forward(input_vector)
                J1 = network.cost(target_output_vector)
                neuron.threshold = old_theta - epsilon
                network.forward(input_vector)
                J2 = network.cost(target_output_vector)
                estimated_gradient = (J1 - J2) / (2*epsilon)
                diff = abs(neuron.threshold_gradient - estimated_gradient)
                assert diff < 0.0001, "t difference: {}".format(diff)
                # print("t difference: {}".format(diff))
                neuron.threshold = old_theta

    def test_reset_gradients(self):
        network = pine.util.create_network([3,5,2], ['logistic']*2)
        for layer in network.layers:
            for neuron in layer.neurons:
                for grad in neuron.weight_gradients:
                    self.assertEqual(grad, 0)
                self.assertEqual(neuron.threshold_gradient, 0)

    def tearDown(self):
        pass


# util.py
class TestUtil(unittest.TestCase):
    """Testing for util"""
    def setUp(self):
        pass

    def test_isValidFunctionName(self):
        self.assertTrue(pine.util.isValidFunction("logistic"))
        self.assertFalse(pine.util.isValidFunction("test"))

    def tearDown(self):
        pass

class TestActivation(unittest.TestCase):
    """Testing for activation"""
    def setUp(self):
        pass

    def test_METHOD(self):
        pass

    def tearDown(self):
        pass

class TestMODULE(unittest.TestCase):
    """Testing for MODULE"""
    def setUp(self):
        pass

    def test_METHOD(self):
        pass

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
