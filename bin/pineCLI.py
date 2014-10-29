#! /usr/bin/env python3
'''
Created on Jul 22, 2013

@author: dusenberrymw
'''
import argparse
import csv
import math
import os.path
import pickle
import random
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

import pine.activation
import pine.data
import pine.network
import pine.trainer
import pine.util

parser = argparse.ArgumentParser(description='Pine: Python Neural Network')
parser.add_argument('examples_file', type=argparse.FileType('r'),
                    help='A txt file containing examples for training \
                          or predicting.\
                          Expects examples file to be of format: \
                          "[target1[,target2[,...]]] | input1[,input2[,...]]"')
parser.add_argument('network_layout', help='number of nodes in each layer \
                                            separated by commas, starting \
                                            with the input layer, and ending \
                                            in the output layer. Ex: 4,10,1 \
                                            is a network with 4 inputs, \
                                            10 nodes in a hidden layer, and 1 \
                                            output node.')
parser.add_argument('-af','--activation_functions', help='activation function names for \
                                                 hidden and output nodes')
parser.add_argument('-l', '--learning_rate', type=float, default=0.01)
parser.add_argument('-m', '--momentum', type=float, default=0.0)
parser.add_argument('-p', '--passes', type=int, default=1)
parser.add_argument('-bs', '--batch_size', type=int)
parser.add_argument('-f', '--model_output', default=None, type=argparse.FileType('wb'))
parser.add_argument('-i', '--model_input', default=None, type=argparse.FileType('rb'))
parser.add_argument('-t', '--testing', action='store_true', default=False)
parser.add_argument('-op', '--only_predict', action='store_true', default=False)
parser.add_argument('-u', '--unsupervised', action='store_true', default=False)
parser.add_argument('-pf', '--predictions_file', default=None, type=argparse.FileType('w'))
parser.add_argument('-np', '--num_processes', type=int, default=None,
                    help='add to limit program to a certain number of processes')
parser.add_argument('-v', '--verbose', action='store_true', default=False)
parser.add_argument('--trainer', default="SGD")

# get args from command line
args = parser.parse_args()

# get data
examples = pine.data.parse_data(args.examples_file, args.only_predict)

# get/make a network
if args.model_input:
    # load in network
    network = pickle.load(args.model_input)
    args.model_input.close()
else:
    # build network
    try:
        network_layout = [int(num_nodes) for num_nodes in
                          args.network_layout.split(",")]
    except ValueError:
        print("\nError: Network layout must be in format: #[,#[,...]],# beginning with input layer, and ending in output layer")
        exit()
    if args.activation_functions: 
        try:
            act_funcs = [f.lower() for f in args.activation_functions.split(",")]
            for f in act_funcs:
                if not pine.util.is_valid_function(f):
                    print("\nError: activation functions must be one of the following: 'logistic', 'tanh', 'linear'")
                    exit()
        except ValueError:
            print("\nError: Activation functions must be in format: name[,name[,...]],name beginning with hidden layer 1, and ending in output layer")
            exit()
    else:
        act_funcs = ['logistic']*(len(network_layout)-1)
    network = pine.util.create_network(network_layout, act_funcs)

# print network structure
if args.verbose:
    print('Number of input neurons: {0}'.format(len(network.layers[0].neurons[0].weights)))
    for i in range(len(network.layers)-1):
        print('Number of hidden neurons: {0}'.format(len(network.layers[i].neurons)))
    print('Number of output neurons: {0}'.format(len(network.layers[-1].neurons)))
    print('Number of total layers: {0}'.format(len(network.layers)+1))

# now determine what needs to be done
if args.only_predict:
    # just predict
    if args.predictions_file:
        writer = csv.writer(args.predictions_file)
    for example in examples:
        hypothesis_vector = network.forward(example[0])
        if args.verbose:
            print('Input: {0}, Target Output: {1}, Actual Output: {2}'.
                format(example[0], example[1], hypothesis_vector))
        if args.predictions_file:
            writer.writerow(hypothesis_vector)

elif args.testing:
    # testing
    if args.predictions_file:
        writer = csv.writer(args.predictions_file)
    for example in examples:
        hypothesis_vector = network.forward(example[0])
        if args.verbose:
            print('Input: {0}, Target Output: {1}, Actual Output: {2}'.
                  format(example[0], example[1], hypothesis_vector))
        if args.predictions_file:
            writer.writerow(hypothesis_vector)
    cost = pine.util.calculate_average_cost(network, examples)
    print('Cost w/ Testing Data: {0}'.format(cost))

else: # train
    if args.batch_size:
        batch_size = args.batch_size
    else:
        batch_size = len(examples)
    print(batch_size)
    # now train on the examples
    if args.trainer.lower() == "backpropagation":
        # older backprop trainer with most logic in the train function
        trainer = pine.trainer.Backpropagation(args.learning_rate, args.momentum)
        pine.trainer.parallel_train(network, trainer, examples, args.passes,
                                args.unsupervised, args.num_processes)
    else:
        # new backprop (SGD) trainer with most logic in the network functions
        trainer = pine.trainer.SGD(args.learning_rate)
        if args.verbose:
            batches = math.ceil(args.passes/batch_size)
            for i in range(batches):
                trainer.train(network, examples, batch_size, batch_size) 
                cost = pine.util.calculate_average_cost(network, examples)
                print('Batch:{}, Cost {}'.format(i+1, cost))
        else:
            trainer.train(network, examples, batch_size, args.passes)

    # and print cost
    cost = pine.util.calculate_average_cost(network, examples)
    print('Cost w/ Training Data: {0}'.format(cost))
    if args.model_output:
        # save the network
        pickle.dump(network, args.model_output)
    elif args.model_input:
        i = open(args.model_input.name, 'wb')
        pickle.dump(network, i)
        i.close()
    if args.predictions_file:
        # write the predictions
        writer = csv.writer(args.predictions_file)
        for example in examples:
            hypothesis_vector = network.forward(example[0])
            writer.writerow(hypothesis_vector)
