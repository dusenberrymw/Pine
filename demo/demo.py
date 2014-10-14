#! /usr/bin/env python3
'''
Created on Jun 6, 2013

@author: dusenberrymw
'''
from multiprocessing import cpu_count
import os.path
import random
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import demo_data
import pine.network
import pine.trainer
import pine.util

# constants for the different projects
AND_PROJECT = 0
XOR_PROJECT = 1
IRIS_PROJECT = 2
LETTER_RECOG_PROJECT = 3

def main(project):
    """Starting point for this program

    Will build a default list of parameters, get the data for the specified
    project, then build and train networks in parallel, trying different
    configurations

    """
    # build the params for the project
    params = build_project_params(project)
    training_data = params['training_data']
    testing_data = params['testing_data']
    iterations = params['iterations']
    num_processes = params['num_processes']

    # create the network
    network = pine.util.create_network(params['num_neurons_list'], params['activation_functions'])
    
    # Test the network
#     network2 = copy.deepcopy(network)
#     test_run(network2, params)

    print()
    for i in range(len(network.layers)-1):
        print('Number of hidden neurons: {0}'.format(len(network.layers[i].neurons)))
    print('Number of output neurons: {0}'.format(len(network.layers[-1].neurons)))
    print('Number of total layers: {0}'.format(len(network.layers)))

    # test the network
    print("\nBefore training")
    pine.util.print_network_error(network, training_data, testing_data)

    # train the network
    trainer = pine.trainer.Backpropagation(params['learning_rate'],
                                       params['momentum_coef'])

    i = 0
#     error = pine.util.calculate_average_cost(network, params['data'].training_inputs,
#                                     params['data'].training_target_outputs)
    while ((i*params['iterations'])<2000):#& (error > params['min_error']):
        pine.trainer.parallel_train(network, trainer, training_data, iterations,
                                num_processes)
#         trainer.train(network, data.training_inputs, data.training_target_outputs, iterations)

        # check the new error on the master network
        print("\nMaster Network:")
        error = pine.util.calculate_average_cost(network, testing_data)
        print('Cost w/ Test Data: {0}'.format(error))
#         pine.util.print_network_error(network, training_data, testing_data)
#         error = pine.util.calculate_average_cost(network, params['data'].training_inputs,
#                                     params['data'].training_target_outputs)
        print("Iteration number: {0}".format((i+1)*params['iterations']))
        i +=1
#         print('{0}\t{1}'.format(params['learning_rate'], params['momentum_coef']))

        # change the learning rate and momentum coef
        params['learning_rate'] += params['learning_rate']*params['learning_rate_change']
#         params['momentum_coef'] += params['momentum_coef']*params['momentum_coef_change']
        trainer.learning_rate = params['learning_rate']
#         trainer.momentum_coef = params['momentum_coef']

    print("\nAfter Training\n")
    pine.util.print_network_outputs(network, testing_data)
    print()
    pine.util.print_network_error(network, training_data, testing_data)
    print('Iterations: {0}'.format(i*params['iterations']))


def build_project_params(project):
    """Create the default parameters for this project"""
    # create a dictionary to store all params
    params = {}

    # Set up default parameters
    params['min_error'] = 0.01
    params['iterations'] = 1000
    params['num_processes'] = None
    params['learning_rate'] = 0.1
    params['momentum_coef'] = 0.5
    params['learning_rate_change'] = 0
    params['momentum_coef_change'] = 0

    # Get the project's data and any overrides to the defaults
    if project == AND_PROJECT:
        params['training_data'], params['testing_data'] =  demo_data.and_data()
        params['activation_functions'] = ['logistic']
        params['num_neurons_list'] = [len(params['training_data'][0][1]), 1] # just single layer perceptron
        params['learning_rate'] = 0.2
        params['iterations'] = 1
        params['num_processes'] = 1
    elif project == XOR_PROJECT:
        params['training_data'], params['testing_data'] = demo_data.xor_data()
        params['activation_functions'] = ['logistic'] * 2
        params['num_neurons_list'] = [2,5,1]
        params['learning_rate'] = 0.1 # 0.1
        params['momentum_coef'] = 0.0
        params['iterations'] = 1
        params['num_processes'] = 1
    elif project == IRIS_PROJECT:
        params['training_data'], params['testing_data'] = demo_data.iris_data()
        params['activation_functions'] = ['logistic'] * 2
        params['num_neurons_list'] = [10, 3]
        params['learning_rate'] = 0.02 #0.007
        params['momentum_coef'] = 0.0 #0.4
        params['learning_rate_change'] = -0.00 #0.2
        params['momentum_coef_change'] = 0.00 #0.2
        params['iterations'] = 1
#         params['num_processes'] = 1
    elif project == LETTER_RECOG_PROJECT:
        params['training_data'], params['testing_data'] = demo_data.letter_recognition_data()
        params['activation_functions'] = ['logistic'] * 2
        params['num_neurons_list'] = [80, 26]
        params['learning_rate'] = 0.2 #0.2
        params['momentum_coef'] = 0.0
        params['learning_rate_change'] = 0.0 #0.2
        params['momentum_coef_change'] = 0.0 #0.2
        params['iterations'] = 1

    return params


def test_run(network, params):
    """Test method for the system

    This creates a network, tests it, trains it, and retests it

    """
    params['iterations'] = 1000

    print('Number of hidden neurons: {0}'.format(len(network.layers[0].neurons)))
    print('Number of output neurons: {0}'.format(len(network.layers[-1].neurons)))
    print('Number of total layers: {0}'.format(len(network.layers)))

    # test the network
    print("\nBefore training")
    pine.util.print_network_error(network, params['data'])

    # Train the network
    print('\nWill train for {0} iterations'.format(params['iterations']))
    trainer = pine.trainer.Backpropagation()
    trainer.train(network, params['data'].training_inputs,
                  params['data'].training_target_outputs,
                  params['learning_rate'], params['momentum_coef'],
                  params['iterations'], params['min_error'])

    # test the network
    print("\nAfter training")
#     pine.util.print_network_outputs(network, params['data'])
    pine.util.print_network_error(network, params['data'])
    print('Trained for {0} iterations'.format(trainer.iterations))


if __name__ == '__main__':
    # this will be used for timing the code
    start_time = time.clock()

    # Choose a project to run
    # 0 = AND_PROJECT
    # 1 = XOR_PROJECT
    # 2 = Iris_PROJECT
    # 3 = LETTER_RECOG_PROJECT
    project = XOR_PROJECT

    if project == AND_PROJECT:
        print("Running AND test project\n")
    elif project == XOR_PROJECT:
        print("Running XOR test project\n")
    elif project == IRIS_PROJECT:
        print("Running Iris test project\n")
    elif project == LETTER_RECOG_PROJECT:
        print("Running Letter Recognition test project\n")

#     import cProfile
#     cProfile.run('main(project)')
    main(project)

    print("\nCode took %s seconds to run\n" % (str(time.clock() - start_time)))
