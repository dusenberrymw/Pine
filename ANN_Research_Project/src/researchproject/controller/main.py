'''
Created on Jun 6, 2013

@author: dusenberrymw
'''

import sys,os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from researchproject.model.network import Network
from researchproject.model import network as network_module
from researchproject.model import training
from researchproject.model import data as data_module
from multiprocessing import cpu_count
import time
import copy

# constants for the different projects
XOR_PROJECT = 1
IRIS_PROJECT = 2
CT_PROJECT = 3
LETTER_RECOG_PROJECT = 4

def main(project):
    """Starting point for this program
    
    Will build a default list of parameters, get the data for the specified
    project, then build and train networks in parallel, trying different
    configurations
    
    """
    # build the params for the project
    params = build_project_params(project)
    data = params['data']
    iterations = params['iterations']
    min_error = params['min_error']
    num_processes = params['num_processes']
    
    # create the network
    network = Network(len(params['data'].training_inputs[0]), 
                      params['num_neurons_list'],
                      params['activation_functions'])
    # Test the network
#     network2 = copy.deepcopy(network)
#     test_run(network2, params)
    
    print('\nNumber of hidden neurons: {0}'.format(len(network.layers[0].neurons)))
    print('Number of output neurons: {0}'.format(len(network.layers[-1].neurons)))
    print('Number of total layers: {0}'.format(len(network.layers)))
    
    # test the network
    print("\nBefore training")
    network_module.print_network_error(network, params['data'])
    
    # train the network
    trainer = training.Backpropagation(params['learning_rate'], 
                                       params['momentum_coef'])
#     trainer = training.ResilientPropagation()
    
    i = 0
    error = network.calculate_error(params['data'].training_inputs, 
                                    params['data'].training_target_outputs)
    while (i<1000000) & (error > params['min_error']):
        training.parallel_train(network, trainer, data.training_inputs, 
                                data.training_target_outputs, iterations,
                                min_error, num_processes)
#         trainer.train(network, data.training_inputs, data.training_target_outputs, iterations, min_error)
        
        # check the new error on the master network
        print("\nMaster Network:")
        network_module.print_network_error(network, params['data'])
        error = network.calculate_error(params['data'].training_inputs, 
                                    params['data'].training_target_outputs)
        i +=1
#         print('{0}\t{1}'.format(params['learning_rate'], params['momentum_coef']))
#           
        # change the learning rate and momentum coef
        params['learning_rate'] += params['learning_rate']*params['learning_rate_change']
        params['momentum_coef'] += params['momentum_coef']*params['momentum_coef_change']
        trainer.learning_rate = params['learning_rate']
        trainer.momentum_coef = params['momentum_coef']
        print(i)
    
    print("\nAfter Training")
    network_module.print_network_error(network, params['data'])
    print('Iterations: {0}'.format(i))
    
    network_module.print_network_outputs(network, params['data'])


def build_project_params(project):
    """Create the default parameters for this project"""
    # create a dictionary to store all params
    params = {}
    
    # Set up default parameters
    params['activation_functions'] = [training.TanhActivationFunction()] * 2
    params['num_neurons_list'] = [None, 1]
    params['min_error'] = 0.01
    params['iterations'] = 1000
    params['num_processes'] = None
    params['learning_rate'] = 0.1
    params['momentum_coef'] = 0.5
    params['learning_rate_change'] = 0
    params['momentum_coef_change'] = 0
    
    # Get the project's data and any overrides to the defaults
    if project == XOR_PROJECT:
        params['data'] = data_module.xor_data()
        params['activation_functions'] = [training.SigmoidActivationFunction()] * 2
        params['num_neurons_list'] = [3,1]
        params['learning_rate'] = 0.5
        params['momentum_coef'] = 0.1
        params['iterations'] = 1
        params['num_processes'] = 1
    elif project == IRIS_PROJECT:
        params['data'] = data_module.iris_data()
        params['learning_rate'] = 0.1 #0.2
        params['momentum_coef'] = 0.1 #0.4
        params['learning_rate_change'] = -0.00 #0.2
        params['momentum_coef_change'] = 0.0 #0.2
        params['iterations'] = 20
#         params['num_processes'] = 1
    elif project == LETTER_RECOG_PROJECT:
        params['data'] = data_module.letter_recognition_data()
        params['learning_rate'] = 0.1
        params['momentum_coef'] = 0.5 
        params['learning_rate_change'] = 0.01 #0.2
        params['momentum_coef_change'] = 0.0 #0.2
        params['iterations'] = 50
    elif project == CT_PROJECT:
        params['data'] = data_module.ct_data()
        params['num_neurons_list'] = [10,1]
        params['activation_functions'] = [training.TanhActivationFunction(), training.SigmoidActivationFunction()]
        params['learning_rate'] = 0.05
        params['momentum_coef'] = 0.5
        params['learning_rate_change'] = -0.01 #0.0
        params['momentum_coef_change'] = 0.01 #0.0
        params['iterations'] = 10 #650 total

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
    network_module.print_network_error(network, params['data'])
    
    # Train the network
    print('\nWill train for {0} iterations'.format(params['iterations']))
    trainer = training.Backpropagation()
    trainer.train(network, params['data'].training_inputs, 
                  params['data'].training_target_outputs, 
                  params['learning_rate'], params['momentum_coef'],
                  params['iterations'], params['min_error'])
    
    # test the network
    print("\nAfter training")
#     network_module.print_network_outputs(network, params['data'])
    network_module.print_network_error(network, params['data'])
    print('Trained for {0} iterations'.format(trainer.iterations))

    
if __name__ == '__main__':
    # this will be used for timing the code
    start_time = time.clock()
    
    # 1 = XOR
    # 2 = Iris
    # 3 = CT
    # 4 = letter recognition
    project = 3
    
    if project == XOR_PROJECT:
        print("Running XOR test project\n")
    elif project == IRIS_PROJECT:
        print("Running Iris test project\n")
    elif project == CT_PROJECT:
        print("Running CT clinical project\n")
    elif project == LETTER_RECOG_PROJECT:
        print("Running Letter Recognition test project\n")
    
#     import cProfile
#     cProfile.run('main(project)')
    main(project)

    print("\nCode took %s seconds to run\n" % (str(time.clock() - start_time)))

    
    