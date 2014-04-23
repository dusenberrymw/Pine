#! /usr/bin/env python3
'''
Created on Jul 22, 2013

@author: dusenberrymw
'''
import argparse
import csv
import os.path
import pickle
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

import pine.data
import pine.network
import pine.training

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
parser.add_argument('-l', '--learning_rate', type=float, default=0.05)
parser.add_argument('-m', '--momentum', type=float, default=0.0)
parser.add_argument('-p', '--passes', type=int, default=1)
parser.add_argument('-f', '--model_output', default=None, type=argparse.FileType('wb'))
parser.add_argument('-i', '--model_input', default=None, type=argparse.FileType('rb'))
parser.add_argument('-t', '--testing', action='store_true', default=False)
parser.add_argument('-op', '--only_predict', action='store_true', default=False)
parser.add_argument('-u', '--unsupervised', action='store_true', default=False)
parser.add_argument('-pf', '--predictions_file', default=None, type=argparse.FileType('w'))
parser.add_argument('-np', '--num_processes', type=int, default=None,
                    help='add to limit program to a certain number of processes')
parser.add_argument('-v', '--verbose', action='store_true', default=False)

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
        num_inputs = network_layout[0]
        network_layout = network_layout[1:]
    except ValueError:
        print("\nError: Network layout must be in format: #[,#[,...]],# beginning with input layer, and ending in output layer")
        exit()
    network = pine.network.Network(num_inputs, network_layout, [pine.training.LogisticActivationFunction()]*len(network_layout))

# print network structure
if args.verbose:
    print('Number of inputs: {0}'.format(len(network.layers[0].neurons[0].weights)))
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
        hypothesis_vector = network.compute_network_output(example[1])
        if args.verbose:
            print('Input: {0}, Target Output: {1}, Actual Output: {2}'.
                format(example[1], example[0], hypothesis_vector))
        if args.predictions_file:
            writer.writerow(hypothesis_vector)

elif args.testing:
    # testing
    if args.predictions_file:
        writer = csv.writer(args.predictions_file)
    for example in examples:
        hypothesis_vector = network.compute_network_output(example[1])
        if args.verbose:
            print('Input: {0}, Target Output: {1}, Actual Output: {2}'.
                  format(example[1], example[0], hypothesis_vector))
        if args.predictions_file:
            writer.writerow(hypothesis_vector)
    error = network.calculate_RMS_error(examples)
    print('Error w/ Testing Data: {0}'.format(error))

else: # train
    # now train on the examples
    trainer = pine.training.Backpropagation(args.learning_rate, args.momentum)
#    for i in range(args.passes):
    pine.training.parallel_train(network, trainer, examples, args.passes,
                            args.unsupervised, args.num_processes)
    # and print error
    error = network.calculate_RMS_error(examples)
    print('Error w/ Training Data: {0}'.format(error))
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
            hypothesis_vector = network.compute_network_output(example[1])
            writer.writerow(hypothesis_vector)
