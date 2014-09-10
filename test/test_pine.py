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

import pine.data
import pine.network
import pine.training
import pine.util

# network.py
class TestNetwork(unittest.TestCase):
    """Testing for network.py"""
    def setUp(self):
        self.activation_func = pine.training.LogisticActivationFunction()
        self.input_vector = [5,6,7]

        self.neuron = pine.network.Neuron(3)
        self.neuron.weights = [1,-2,3]
        self.neuron.threshold = 4

        local_output = sum([x*y for x,y in zip(self.input_vector, self.neuron.weights)]) + self.neuron.threshold
        self.output = 1.0 / (1 + math.exp(-1.0*local_output)) #0.99999999999

        self.layer = pine.network.Layer(2, 3, self.activation_func)
        self.layer.neurons = [self.neuron, self.neuron]

    def test_neuron_forward(self):
        self.assertEqual(self.neuron.forward(self.input_vector,
                                             self.activation_func), self.output)

    def test_layer_forward(self):
        self.assertEqual(self.layer.forward(self.input_vector),
                         [self.output, self.output])

    def test_network_forward(self):
        network = pine.network.Network()
        network.layers.append(self.layer)
        new_neuron = pine.network.Neuron(2)
        new_neuron.weights = [1,-2]
        new_neuron.threshold = 4
        new_layer = pine.network.Layer(2, 3, self.activation_func)
        new_layer.neurons = [self.neuron]
        network.layers.append(new_layer)

        local_output = sum([x*y for x,y in zip([self.output, self.output], self.neuron.weights)]) + self.neuron.threshold
        out = [1.0 / (1 + math.exp(-1.0*local_output))]
        self.assertEqual(network.forward(self.input_vector), out) #0.9525741275104728

    def tearDown(self):
        pass


# util.py
class TestUtil(unittest.TestCase):
    """Testing for util"""
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
